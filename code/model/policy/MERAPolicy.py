import numpy as np
import torch
import torch.nn as nn

from model.components import DNN
from model.policy.OneStagePolicy import OneStagePolicy


class MERAPolicy(OneStagePolicy):
    @staticmethod
    def parse_model_args(parser):
        parser = OneStagePolicy.parse_model_args(parser)
        parser.add_argument(
            "--mera_shortlist_size",
            type=int,
            default=50,
            help="shortlist size K in MERA",
        )
        parser.add_argument(
            "--mera_policy_hidden",
            type=int,
            nargs="+",
            default=[256, 64],
            help="hidden dims for MERA action heads",
        )
        parser.add_argument(
            "--mera_sinkhorn_tau",
            type=float,
            default=0.7,
            help="sinkhorn temperature",
        )
        parser.add_argument(
            "--mera_sinkhorn_iter",
            type=int,
            default=8,
            help="number of sinkhorn iterations",
        )
        parser.add_argument(
            "--mera_gumbel_noise",
            type=float,
            default=0.0,
            help="gumbel noise scale for matrix logits",
        )
        parser.add_argument(
            "--mera_dump_bias",
            type=float,
            default=0.0,
            help="bias term on dump slot logit",
        )
        return parser

    def __init__(self, args, env):
        self.sinkhorn_tau = args.mera_sinkhorn_tau
        self.sinkhorn_iter = args.mera_sinkhorn_iter
        self.gumbel_noise = args.mera_gumbel_noise
        self.shortlist_size = min(args.mera_shortlist_size, env.n_candidate)
        if self.shortlist_size < env.slate_size:
            raise ValueError(
                f"mera_shortlist_size({self.shortlist_size}) must be >= slate_size({env.slate_size})"
            )
        super().__init__(args, env)
        self.n_slot = env.slate_size
        self.matrix_cols = self.n_slot + 1

        self.hyper_action_dim = self.shortlist_size * self.matrix_cols
        self.action_dim = self.hyper_action_dim
        self.effect_action_dim = self.n_slot

        self.shortlist_query_layer = DNN(
            self.state_dim,
            args.mera_policy_hidden,
            self.enc_dim,
            dropout_rate=self.dropout_rate,
            do_batch_norm=True,
        )
        self.slot_query_layer = DNN(
            self.state_dim,
            args.mera_policy_hidden,
            self.n_slot * self.enc_dim,
            dropout_rate=self.dropout_rate,
            do_batch_norm=True,
        )
        self.dump_query_layer = DNN(
            self.state_dim,
            args.mera_policy_hidden,
            self.enc_dim,
            dropout_rate=self.dropout_rate,
            do_batch_norm=True,
        )
        self.dump_bias = nn.Parameter(torch.tensor(float(args.mera_dump_bias)))

    def sinkhorn_partial(self, logits):
        """
        logits: (B, K, N+1), with K >= N.
        Output soft matrix with row-sum 1 and column sums [1,...,1,K-N].
        """
        B, K, N1 = logits.shape
        matrix = torch.exp(logits / max(self.sinkhorn_tau, 1e-6)).clamp_min(1e-8)

        row_target = torch.ones(B, K, device=logits.device)
        col_target = torch.ones(B, N1, device=logits.device)
        col_target[:, -1] = float(max(K - self.n_slot, 0))

        for _ in range(self.sinkhorn_iter):
            matrix = matrix / matrix.sum(dim=2, keepdim=True).clamp_min(1e-8)
            matrix = matrix * row_target.unsqueeze(-1)
            matrix = matrix / matrix.sum(dim=1, keepdim=True).clamp_min(1e-8)
            matrix = matrix * col_target.unsqueeze(1)

        return matrix

    def greedy_project(self, slot_scores):
        """
        slot_scores: (B, K, N) -> selected row indices in shortlist, shape (B, N).
        """
        B, K, N = slot_scores.shape
        selected_rows = torch.zeros(B, N, dtype=torch.long, device=slot_scores.device)
        used_mask = torch.zeros(B, K, dtype=torch.bool, device=slot_scores.device)
        very_small = torch.tensor(-1e9, device=slot_scores.device)

        for slot in range(N):
            scores = slot_scores[:, :, slot]
            scores = torch.where(used_mask, very_small, scores)
            picked = torch.argmax(scores, dim=1)
            selected_rows[:, slot] = picked
            used_mask.scatter_(1, picked.view(B, 1), True)
        return selected_rows

    def generate_action(self, state_dict, feed_dict):
        user_state = state_dict["state"]
        candidates = feed_dict["candidates"]
        epsilon = feed_dict["epsilon"]
        do_explore = feed_dict["do_explore"]
        batch_wise = feed_dict["batch_wise"]

        B = user_state.shape[0]
        do_uniform_shortlist = do_explore and (np.random.random() < epsilon)

        candidate_item_enc, reg = self.user_encoder.get_item_encoding(
            candidates["item_id"],
            {k[3:]: v for k, v in candidates.items() if k != "item_id"},
            B if batch_wise else 1,
        )

        if batch_wise:
            candidate_ids = candidates["item_id"].view(B, -1)
            candidate_item_enc = candidate_item_enc.view(B, -1, self.enc_dim)
        else:
            candidate_ids = candidates["item_id"].view(1, -1).expand(B, -1)
            candidate_item_enc = candidate_item_enc.view(1, -1, self.enc_dim).expand(B, -1, -1)

        L = candidate_item_enc.shape[1]

        shortlist_query = self.shortlist_query_layer(user_state).view(B, 1, self.enc_dim)
        shortlist_scores = torch.sum(shortlist_query * candidate_item_enc, dim=-1)

        if do_uniform_shortlist:
            shortlist_indices = torch.topk(
                torch.rand(B, L, device=user_state.device),
                k=self.shortlist_size,
                dim=1,
            ).indices
        else:
            shortlist_indices = torch.topk(shortlist_scores, k=self.shortlist_size, dim=1).indices

        shortlist_item_enc = torch.gather(
            candidate_item_enc,
            1,
            shortlist_indices.unsqueeze(-1).expand(-1, -1, self.enc_dim),
        )

        slot_query = self.slot_query_layer(user_state).view(B, self.n_slot, self.enc_dim)
        slot_logits = torch.einsum("bkd,bnd->bkn", shortlist_item_enc, slot_query)

        dump_query = self.dump_query_layer(user_state).view(B, 1, self.enc_dim)
        dump_logits = torch.sum(shortlist_item_enc * dump_query, dim=-1, keepdim=True) + self.dump_bias
        matrix_logits = torch.cat([slot_logits, dump_logits], dim=2)

        if do_explore and self.gumbel_noise > 0:
            uniform_noise = torch.rand_like(matrix_logits).clamp(1e-8, 1.0 - 1e-8)
            gumbel_noise = -torch.log(-torch.log(uniform_noise))
            matrix_logits = matrix_logits + self.gumbel_noise * gumbel_noise

        soft_matrix = self.sinkhorn_partial(matrix_logits)
        slot_probs = soft_matrix[:, :, : self.n_slot]

        selected_rows = self.greedy_project(slot_probs)
        selected_candidate_indices = torch.gather(shortlist_indices, 1, selected_rows)
        selected_item_ids = torch.gather(candidate_ids, 1, selected_candidate_indices)
        selected_scores = torch.gather(shortlist_scores, 1, selected_candidate_indices)

        matrix_entropy = -(
            soft_matrix.clamp_min(1e-8) * torch.log(soft_matrix.clamp_min(1e-8))
        ).sum(dim=(1, 2)) / float(self.shortlist_size * self.matrix_cols)

        reg = (
            reg
            + self.get_regularization(
                self.shortlist_query_layer, self.slot_query_layer, self.dump_query_layer
            )
            + self.dump_bias * self.dump_bias
        )

        flat_action = soft_matrix.view(B, -1)
        return {
            "preds": selected_scores,
            "indices": selected_candidate_indices,
            "action": flat_action,
            "hyper_action": flat_action,
            "mera_action": flat_action,
            "effect_action": selected_item_ids,
            "all_preds": shortlist_scores,
            "shortlist_indices": shortlist_indices,
            "mera_matrix": soft_matrix,
            "mera_entropy": matrix_entropy,
            "reg": reg,
        }

    def forward(self, feed_dict: dict, return_prob=True):
        out_dict = self.get_forward(feed_dict)
        if return_prob:
            out_dict["all_probs"] = torch.softmax(out_dict["all_preds"], dim=1)
            out_dict["probs"] = out_dict["preds"]
        return out_dict
