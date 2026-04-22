import torch
import torch.nn as nn
import torch.nn.functional as F

from model.components import DNN
from utils import get_regularization


class MERACritic(nn.Module):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument(
            "--critic_hidden_dims",
            type=int,
            nargs="+",
            default=[256, 64],
            help="hidden layers for MERA critic",
        )
        parser.add_argument(
            "--critic_dropout_rate",
            type=float,
            default=0.1,
            help="dropout rate in MERA critic",
        )
        return parser

    def __init__(self, args, environment, policy):
        super().__init__()
        self.state_dim = policy.state_dim
        self.action_dim = policy.action_dim
        self.net = DNN(
            self.state_dim + self.action_dim,
            args.critic_hidden_dims,
            1,
            dropout_rate=args.critic_dropout_rate,
            do_batch_norm=True,
        )

    def forward(self, feed_dict):
        state_emb = feed_dict["state"].view(-1, self.state_dim)
        action_emb = feed_dict["action"].view(-1, self.action_dim)
        q = self.net(torch.cat((state_emb, action_emb), dim=-1)).view(-1)
        reg = get_regularization(self.net)
        return {"q": q, "reg": reg}
