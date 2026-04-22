import numpy as np
import torch

from model.buffer.BaseBuffer import BaseBuffer


class MERABuffer(BaseBuffer):
    @staticmethod
    def parse_model_args(parser):
        parser = BaseBuffer.parse_model_args(parser)
        return parser

    def reset(self, *reset_args):
        env = reset_args[0]
        actor = reset_args[1]
        super().reset(env, actor)

        self.buffer["user_response"]["immediate_response"] = torch.zeros(
            self.buffer_size, env.response_dim * actor.effect_action_dim
        ).to(torch.float).to(self.device)

        self.buffer["policy_output"]["action"] = torch.zeros(
            self.buffer_size, actor.action_dim
        ).to(torch.float).to(self.device)
        self.buffer["policy_output"]["effect_action"] = torch.zeros(
            self.buffer_size, actor.effect_action_dim
        ).to(torch.long).to(self.device)
        self.buffer["policy_output"]["entropy"] = torch.zeros(self.buffer_size).to(torch.float).to(self.device)
        return self.buffer

    def sample(self, batch_size):
        indices = np.random.randint(0, self.current_buffer_size, size=batch_size)

        profile = {k: v[indices] for k, v in self.buffer["observation"]["user_profile"].items()}
        history = {k: v[indices] for k, v in self.buffer["observation"]["user_history"].items()}
        observation = {"user_profile": profile, "user_history": history}

        profile = {k: v[indices] for k, v in self.buffer["next_observation"]["user_profile"].items()}
        history = {k: v[indices] for k, v in self.buffer["next_observation"]["user_history"].items()}
        next_observation = {"user_profile": profile, "user_history": history}

        action = self.buffer["policy_output"]["action"][indices]
        policy_output = {
            "state": self.buffer["policy_output"]["state"][indices],
            "action": action,
            "hyper_action": action,
            "mera_action": action,
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
        B = len(user_feedback["reward"])
        if self.buffer_head + B >= self.buffer_size:
            tail = self.buffer_size - self.buffer_head
            indices = [self.buffer_head + i for i in range(tail)] + [i for i in range(B - tail)]
        else:
            indices = [self.buffer_head + i for i in range(B)]
        indices = torch.tensor(indices).to(torch.long).to(self.device)

        for k, v in observation["user_profile"].items():
            self.buffer["observation"]["user_profile"][k][indices] = v
        for k, v in observation["user_history"].items():
            self.buffer["observation"]["user_history"][k][indices] = v

        for k, v in next_observation["user_profile"].items():
            self.buffer["next_observation"]["user_profile"][k][indices] = v
        for k, v in next_observation["user_history"].items():
            self.buffer["next_observation"]["user_history"][k][indices] = v

        action = policy_output.get("mera_action", policy_output["action"])
        self.buffer["policy_output"]["state"][indices] = policy_output["state"]
        self.buffer["policy_output"]["action"][indices] = action
        self.buffer["policy_output"]["effect_action"][indices] = policy_output["effect_action"]
        entropy = policy_output.get("mera_entropy", torch.zeros(B, device=self.device))
        self.buffer["policy_output"]["entropy"][indices] = entropy.view(-1)

        self.buffer["user_response"]["immediate_response"][indices] = user_feedback["immediate_response"].view(B, -1)
        self.buffer["user_response"]["reward"][indices] = user_feedback["reward"]
        self.buffer["done_mask"][indices] = user_feedback["done"]

        self.buffer_head = (self.buffer_head + B) % self.buffer_size
        self.n_stream_record += B
        self.current_buffer_size = min(self.n_stream_record, self.buffer_size)
