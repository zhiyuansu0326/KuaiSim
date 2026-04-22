import torch
import torch.nn as nn

from model.components import DNN
from utils import get_regularization


class MERACompleteCritic(nn.Module):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument(
            "--merac_critic_hidden_dims",
            type=int,
            nargs="+",
            default=[256, 64],
            help="hidden layers for MERAComplete critic output MLP",
        )
        parser.add_argument(
            "--merac_critic_dropout_rate",
            type=float,
            default=0.1,
            help="dropout rate in MERAComplete critic",
        )
        parser.add_argument(
            "--merac_critic_transformer_n_head",
            type=int,
            default=4,
            help="number of attention heads in ordered-slate transformer",
        )
        parser.add_argument(
            "--merac_critic_transformer_d_forward",
            type=int,
            default=64,
            help="feedforward hidden dimension in ordered-slate transformer",
        )
        parser.add_argument(
            "--merac_critic_transformer_n_layer",
            type=int,
            default=2,
            help="number of transformer layers for ordered slate encoding",
        )
        return parser

    def __init__(self, args, environment, policy):
        super().__init__()
        if not hasattr(environment, "candidate_item_encoding"):
            raise ValueError(
                "MERACompleteCritic requires environment.candidate_item_encoding for ordered-slate construction."
            )

        self.state_dim = policy.state_dim
        self.shortlist_size = policy.shortlist_size
        self.n_slot = environment.slate_size
        self.matrix_cols = self.n_slot + 1

        candidate_item_encoding = environment.candidate_item_encoding.detach().clone().to(args.device)
        self.register_buffer("candidate_item_encoding", candidate_item_encoding)
        self.item_enc_dim = candidate_item_encoding.shape[1]

        n_head = args.merac_critic_transformer_n_head
        if self.item_enc_dim % n_head != 0:
            raise ValueError(
                f"item_enc_dim ({self.item_enc_dim}) must be divisible by "
                f"merac_critic_transformer_n_head ({n_head})."
            )

        self.pos_emb = nn.Embedding(self.n_slot, self.item_enc_dim)
        self.register_buffer("pos_index", torch.arange(self.n_slot, dtype=torch.long))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.item_enc_dim,
            dim_feedforward=args.merac_critic_transformer_d_forward,
            nhead=n_head,
            dropout=args.merac_critic_dropout_rate,
            batch_first=True,
        )
        self.transformer_ord = nn.TransformerEncoder(
            encoder_layer, num_layers=args.merac_critic_transformer_n_layer
        )
        self.q_net = DNN(
            self.state_dim + self.item_enc_dim,
            args.merac_critic_hidden_dims,
            1,
            dropout_rate=args.merac_critic_dropout_rate,
            do_batch_norm=True,
        )

    def forward(self, feed_dict):
        """
        feed_dict:
        - state: (B, state_dim)
        - action_matrix: (B, K, N+1) or (B, K*(N+1))
        - shortlist_indices: (B, K)
        """
        state_emb = feed_dict["state"].reshape(-1, self.state_dim)

        action_matrix = feed_dict["action_matrix"]
        if action_matrix.dim() == 2:
            action_matrix = action_matrix.reshape(-1, self.shortlist_size, self.matrix_cols)
        action_matrix = action_matrix.to(state_emb.dtype)

        shortlist_indices = feed_dict["shortlist_indices"].reshape(-1, self.shortlist_size).long()
        shortlist_item_enc = self.candidate_item_encoding[shortlist_indices]

        slot_weights = action_matrix[:, :, : self.n_slot]
        ordered_slate = torch.bmm(slot_weights.transpose(1, 2), shortlist_item_enc)

        pos_emb = self.pos_emb(self.pos_index).reshape(1, self.n_slot, self.item_enc_dim)
        transformer_out = self.transformer_ord(ordered_slate + pos_emb)
        pooled_ord = transformer_out.mean(dim=1)

        q_input = torch.cat((state_emb, pooled_ord), dim=-1)
        q = self.q_net(q_input).view(-1)

        reg = get_regularization(self.q_net, self.pos_emb, self.transformer_ord)
        return {"q": q, "reg": reg, "ordered_slate": ordered_slate}
