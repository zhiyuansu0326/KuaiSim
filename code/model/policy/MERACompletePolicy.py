import numpy as np
import torch
import torch.nn as nn

from model.components import DNN
from model.policy.OneStagePolicy import OneStagePolicy

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None


class MERACompletePolicy(OneStagePolicy):
    @staticmethod
    def parse_model_args(parser):
        parser = OneStagePolicy.parse_model_args(parser)
        parser.add_argument(
            "--merac_shortlist_size",
            type=int,
            default=50,
            help="shortlist size K in MERAComplete",
        )
        parser.add_argument(
            "--merac_policy_hidden",
            type=int,
            nargs="+",
            default=[256, 64],
            help="hidden dims for MERAComplete policy heads",
        )
        parser.add_argument(
            "--merac_tau",
            type=float,
            default=0.7,
            help="Sinkhorn temperature and matrix-entropy coefficient",
        )
        parser.add_argument(
            "--merac_sinkhorn_iter",
            type=int,
            default=8,
            help="number of Sinkhorn iterations",
        )
        parser.add_argument(
            "--merac_gumbel_noise",
            type=float,
            default=1.0,
            help="Gumbel perturbation scale",
        )
        parser.add_argument(
            "--merac_dump_bias",
            type=float,
            default=0.0,
            help="bias term on dump-slot logit",
        )
        return parser

    def __init__(self, args, env):
        if linear_sum_assignment is None:
            raise ImportError(
                "MERACompletePolicy requires scipy.optimize.linear_sum_assignment for Hungarian rounding."
            )
        self.sinkhorn_tau = args.merac_tau
        self.sinkhorn_iter = args.merac_sinkhorn_iter
        self.gumbel_noise = args.merac_gumbel_noise
        self.shortlist_size = min(args.merac_shortlist_size, env.n_candidate)
        if self.shortlist_size < env.slate_size:
            raise ValueError(
                f"merac_shortlist_size({self.shortlist_size}) must be >= slate_size({env.slate_size})"
            )
        super().__init__(args, env)
        self.n_slot = env.slate_size
        self.matrix_cols = self.n_slot + 1
        self.dump_slot_idx = self.n_slot

        self.hyper_action_dim = self.shortlist_size * self.matrix_cols
        self.action_dim = self.hyper_action_dim
        self.effect_action_dim = self.n_slot

        self.shortlist_query_layer = DNN(
            self.state_dim,
            args.merac_policy_hidden,
            self.enc_dim,
            dropout_rate=self.dropout_rate,
            do_batch_norm=True,
        )
        self.slot_query_layer = DNN(
            self.state_dim,
            args.merac_policy_hidden,
            self.n_slot * self.enc_dim,
            dropout_rate=self.dropout_rate,
            do_batch_norm=True,
        )
        self.dump_query_layer = DNN(
            self.state_dim,
            args.merac_policy_hidden,
            self.enc_dim,
            dropout_rate=self.dropout_rate,
            do_batch_norm=True,
        )
        self.dump_bias = nn.Parameter(torch.tensor(float(args.merac_dump_bias)))

    def sinkhorn_partial(self, logits):
        """
        logits: (B, K, N+1), with K >= N.
        Output matrix with row sums 1 and column sums [1,...,1,K-N].
        """
        batch_size, shortlist_size, n_col = logits.shape
        matrix = torch.exp(logits / max(self.sinkhorn_tau, 1e-6)).clamp_min(1e-8)

        row_target = torch.ones(batch_size, shortlist_size, device=logits.device)
        col_target = torch.ones(batch_size, n_col, device=logits.device)
        col_target[:, -1] = float(max(shortlist_size - self.n_slot, 0))

        for _ in range(self.sinkhorn_iter):
            matrix = matrix / matrix.sum(dim=2, keepdim=True).clamp_min(1e-8)
            matrix = matrix * row_target.unsqueeze(-1)
            matrix = matrix / matrix.sum(dim=1, keepdim=True).clamp_min(1e-8)
            matrix = matrix * col_target.unsqueeze(1)

        return matrix

    def _hungarian_round(self, soft_matrix):
        """
        soft_matrix: (B, K, N+1)
        Returns:
        - hard_matrix: (B, K, N+1)
        - selected_rows: (B, N), row index selected for each real slot
        """
        batch_size, shortlist_size, _ = soft_matrix.shape
        dump_capacity = max(shortlist_size - self.n_slot, 0)
        hard_matrix = torch.zeros_like(soft_matrix)
        selected_rows = torch.zeros(
            batch_size, self.n_slot, dtype=torch.long, device=soft_matrix.device
        )

        for b in range(batch_size):
            score = soft_matrix[b]
            if dump_capacity > 0:
                expanded = torch.cat(
                    [
                        score[:, : self.n_slot],
                        score[:, self.dump_slot_idx : self.dump_slot_idx + 1].expand(
                            shortlist_size, dump_capacity
                        ),
                    ],
                    dim=1,
                )
            else:
                expanded = score[:, : self.n_slot]

            row_ind, col_ind = linear_sum_assignment(
                -expanded.detach().cpu().numpy(),
            )
            row_ind = torch.tensor(row_ind, dtype=torch.long, device=soft_matrix.device)
            col_ind = torch.tensor(col_ind, dtype=torch.long, device=soft_matrix.device)

            hard_b = torch.zeros(
                shortlist_size,
                self.matrix_cols,
                dtype=soft_matrix.dtype,
                device=soft_matrix.device,
            )
            real_slot_mask = col_ind < self.n_slot
            dump_slot_mask = torch.logical_not(real_slot_mask)
            if torch.any(real_slot_mask):
                hard_b[row_ind[real_slot_mask], col_ind[real_slot_mask]] = 1.0
            if torch.any(dump_slot_mask):
                hard_b[row_ind[dump_slot_mask], self.dump_slot_idx] = 1.0

            slot_to_row = torch.full(
                (self.n_slot,),
                -1,
                dtype=torch.long,
                device=soft_matrix.device,
            )
            if torch.any(real_slot_mask):
                slot_to_row[col_ind[real_slot_mask]] = row_ind[real_slot_mask]

            if torch.any(slot_to_row < 0):
                used = torch.zeros(shortlist_size, dtype=torch.bool, device=soft_matrix.device)
                valid_rows = slot_to_row[slot_to_row >= 0]
                if valid_rows.numel() > 0:
                    used[valid_rows] = True
                missing_slots = torch.where(slot_to_row < 0)[0]
                for slot in missing_slots:
                    slot_score = score[:, slot]
                    slot_score = torch.where(
                        used,
                        torch.full_like(slot_score, -1e9),
                        slot_score,
                    )
                    pick = torch.argmax(slot_score)
                    slot_to_row[slot] = pick
                    used[pick] = True
                    hard_b[pick, slot] = 1.0
                    hard_b[pick, self.dump_slot_idx] = 0.0

            hard_matrix[b] = hard_b
            selected_rows[b] = slot_to_row

        return hard_matrix, selected_rows

    def generate_action(self, state_dict, feed_dict):
        user_state = state_dict["state"]
        candidates = feed_dict["candidates"]
        epsilon = feed_dict["epsilon"]
        do_explore = feed_dict["do_explore"]
        batch_wise = feed_dict["batch_wise"]
        is_train = feed_dict["is_train"]

        batch_size = user_state.shape[0]
        do_uniform_shortlist = do_explore and (np.random.random() < epsilon)

        candidate_item_enc, reg = self.user_encoder.get_item_encoding(
            candidates["item_id"],
            {k[3:]: v for k, v in candidates.items() if k != "item_id"},
            batch_size if batch_wise else 1,
        )

        if batch_wise:
            candidate_ids = candidates["item_id"].view(batch_size, -1)
            candidate_item_enc = candidate_item_enc.view(batch_size, -1, self.enc_dim)
        else:
            candidate_ids = candidates["item_id"].view(1, -1).expand(batch_size, -1)
            candidate_item_enc = candidate_item_enc.view(1, -1, self.enc_dim).expand(
                batch_size, -1, -1
            )

        n_candidate = candidate_item_enc.shape[1]

        shortlist_query = self.shortlist_query_layer(user_state).view(batch_size, 1, self.enc_dim)
        shortlist_scores = torch.sum(shortlist_query * candidate_item_enc, dim=-1)

        if do_uniform_shortlist:
            shortlist_indices = torch.topk(
                torch.rand(batch_size, n_candidate, device=user_state.device),
                k=self.shortlist_size,
                dim=1,
            ).indices
        else:
            shortlist_indices = torch.topk(
                shortlist_scores, k=self.shortlist_size, dim=1
            ).indices

        shortlist_item_enc = torch.gather(
            candidate_item_enc,
            1,
            shortlist_indices.unsqueeze(-1).expand(-1, -1, self.enc_dim),
        )

        slot_query = self.slot_query_layer(user_state).view(batch_size, self.n_slot, self.enc_dim)
        slot_logits = torch.einsum("bkd,bnd->bkn", shortlist_item_enc, slot_query)

        dump_query = self.dump_query_layer(user_state).view(batch_size, 1, self.enc_dim)
        dump_logits = torch.sum(shortlist_item_enc * dump_query, dim=-1, keepdim=True) + self.dump_bias
        logits = torch.cat([slot_logits, dump_logits], dim=2)

        if self.gumbel_noise > 0 and (is_train or do_explore):
            uniform_noise = torch.rand_like(logits).clamp(1e-8, 1.0 - 1e-8)
            sampled_gumbel = -torch.log(-torch.log(uniform_noise))
            gumbel_noise = sampled_gumbel * self.gumbel_noise
        else:
            gumbel_noise = torch.zeros_like(logits)

        soft_matrix = self.sinkhorn_partial(logits + gumbel_noise)
        hard_matrix, selected_rows = self._hungarian_round(soft_matrix)

        selected_candidate_indices = torch.gather(shortlist_indices, 1, selected_rows)
        selected_item_ids = torch.gather(candidate_ids, 1, selected_candidate_indices)
        selected_scores = torch.gather(shortlist_scores, 1, selected_candidate_indices)

        matrix_entropy = -(
            soft_matrix.clamp_min(1e-8) * torch.log(soft_matrix.clamp_min(1e-8))
        ).sum(dim=(1, 2)) / float(self.shortlist_size * self.matrix_cols)

        reg = (
            reg
            + self.get_regularization(
                self.shortlist_query_layer,
                self.slot_query_layer,
                self.dump_query_layer,
            )
            + self.dump_bias * self.dump_bias
        )

        hard_flat = hard_matrix.reshape(batch_size, -1)
        soft_flat = soft_matrix.reshape(batch_size, -1)

        return {
            "preds": selected_scores,
            "indices": selected_candidate_indices,
            "action": hard_flat,
            "hyper_action": hard_flat,
            "merac_hard_action": hard_flat,
            "merac_soft_action": soft_flat,
            "effect_action": selected_item_ids,
            "all_preds": shortlist_scores,
            "shortlist_indices": shortlist_indices,
            "soft_action_matrix": soft_matrix,
            "hard_action_matrix": hard_matrix,
            "logits": logits,
            "gumbel_noise": gumbel_noise,
            "mera_entropy": matrix_entropy,
            "reg": reg,
        }

    def forward(self, feed_dict: dict, return_prob=True):
        out_dict = self.get_forward(feed_dict)
        if return_prob:
            out_dict["all_probs"] = torch.softmax(out_dict["all_preds"], dim=1)
            out_dict["probs"] = out_dict["preds"]
        return out_dict
