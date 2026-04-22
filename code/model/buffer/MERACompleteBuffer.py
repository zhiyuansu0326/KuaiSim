import numpy as np
import torch

from model.buffer.BaseBuffer import BaseBuffer


class MERACompleteBuffer(BaseBuffer):
    @staticmethod
    def parse_model_args(parser):
        parser = BaseBuffer.parse_model_args(parser)
        return parser

    def reset(self, *reset_args):
        env = reset_args[0]
        actor = reset_args[1]
        super().reset(env, actor)

        self.shortlist_size = actor.shortlist_size
        self.matrix_cols = actor.matrix_cols
        self.effect_action_dim = actor.effect_action_dim

        self.buffer["user_response"]["immediate_response"] = torch.zeros(
            self.buffer_size, env.response_dim * self.effect_action_dim
        ).to(torch.float).to(self.device)

        self.buffer["policy_output"]["action"] = torch.zeros(
            self.buffer_size, actor.action_dim
        ).to(torch.float).to(self.device)
        self.buffer["policy_output"]["hard_action_matrix"] = torch.zeros(
            self.buffer_size, actor.action_dim
        ).to(torch.float).to(self.device)
        self.buffer["policy_output"]["shortlist_indices"] = torch.zeros(
            self.buffer_size, self.shortlist_size
        ).to(torch.long).to(self.device)
        self.buffer["policy_output"]["indices"] = torch.zeros(
            self.buffer_size, self.effect_action_dim
        ).to(torch.long).to(self.device)
        self.buffer["policy_output"]["effect_action"] = torch.zeros(
            self.buffer_size, self.effect_action_dim
        ).to(torch.long).to(self.device)
        self.buffer["policy_output"]["entropy"] = torch.zeros(self.buffer_size).to(torch.float).to(
            self.device
        )
        return self.buffer

    def sample(self, batch_size):
        indices = np.random.randint(0, self.current_buffer_size, size=batch_size)

        profile = {k: v[indices] for k, v in self.buffer["observation"]["user_profile"].items()}
        history = {k: v[indices] for k, v in self.buffer["observation"]["user_history"].items()}
        observation = {"user_profile": profile, "user_history": history}

        profile = {
            k: v[indices] for k, v in self.buffer["next_observation"]["user_profile"].items()
        }
        history = {
            k: v[indices] for k, v in self.buffer["next_observation"]["user_history"].items()
        }
        next_observation = {"user_profile": profile, "user_history": history}

        hard_action_flat = self.buffer["policy_output"]["hard_action_matrix"][indices]
        policy_output = {
            "state": self.buffer["policy_output"]["state"][indices],
            "action": hard_action_flat,
            "hyper_action": hard_action_flat,
            "merac_hard_action": hard_action_flat,
            "hard_action_matrix": hard_action_flat.reshape(
                -1, self.shortlist_size, self.matrix_cols
            ),
            "shortlist_indices": self.buffer["policy_output"]["shortlist_indices"][indices],
            "indices": self.buffer["policy_output"]["indices"][indices],
            "effect_action": self.buffer["policy_output"]["effect_action"][indices],
            "mera_entropy": self.buffer["policy_output"]["entropy"][indices],
        }

        user_response = {
            "reward": self.buffer["user_response"]["reward"][indices],
            "immediate_response": self.buffer["user_response"]["immediate_response"][indices],
        }
        done_mask = self.buffer["done_mask"][indices]
        return observation, policy_output, user_response, done_mask, next_observation

    def update(self, observation, policy_output, user_feedback, next_observation):
        batch_size = len(user_feedback["reward"])
        if self.buffer_head + batch_size >= self.buffer_size:
            tail = self.buffer_size - self.buffer_head
            indices = [self.buffer_head + i for i in range(tail)] + [
                i for i in range(batch_size - tail)
            ]
        else:
            indices = [self.buffer_head + i for i in range(batch_size)]
        indices = torch.tensor(indices).to(torch.long).to(self.device)

        for k, v in observation["user_profile"].items():
            self.buffer["observation"]["user_profile"][k][indices] = v
        for k, v in observation["user_history"].items():
            self.buffer["observation"]["user_history"][k][indices] = v

        for k, v in next_observation["user_profile"].items():
            self.buffer["next_observation"]["user_profile"][k][indices] = v
        for k, v in next_observation["user_history"].items():
            self.buffer["next_observation"]["user_history"][k][indices] = v

        hard_action = policy_output.get("hard_action_matrix", policy_output["action"])
        if hard_action.dim() == 3:
            hard_action = hard_action.reshape(batch_size, -1)

        self.buffer["policy_output"]["state"][indices] = policy_output["state"]
        self.buffer["policy_output"]["action"][indices] = hard_action
        self.buffer["policy_output"]["hard_action_matrix"][indices] = hard_action
        self.buffer["policy_output"]["shortlist_indices"][indices] = policy_output["shortlist_indices"]
        self.buffer["policy_output"]["indices"][indices] = policy_output["indices"]
        self.buffer["policy_output"]["effect_action"][indices] = policy_output["effect_action"]
        entropy = policy_output.get("mera_entropy", torch.zeros(batch_size, device=self.device))
        self.buffer["policy_output"]["entropy"][indices] = entropy.view(-1)

        self.buffer["user_response"]["immediate_response"][indices] = user_feedback[
            "immediate_response"
        ].view(batch_size, -1)
        self.buffer["user_response"]["reward"][indices] = user_feedback["reward"]
        self.buffer["done_mask"][indices] = user_feedback["done"]

        self.buffer_head = (self.buffer_head + batch_size) % self.buffer_size
        self.n_stream_record += batch_size
        self.current_buffer_size = min(self.n_stream_record, self.buffer_size)
