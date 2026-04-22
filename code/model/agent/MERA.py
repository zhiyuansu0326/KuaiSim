import copy
import torch
import torch.nn.functional as F

from model.agent.BaseRLAgent import BaseRLAgent


class MERA(BaseRLAgent):
    @staticmethod
    def parse_model_args(parser):
        parser = BaseRLAgent.parse_model_args(parser)
        parser.add_argument("--critic_lr", type=float, default=1e-4, help="critic learning rate")
        parser.add_argument("--critic_decay", type=float, default=1e-4, help="critic weight decay")
        parser.add_argument(
            "--target_mitigate_coef",
            type=float,
            default=0.01,
            help="target network update coefficient",
        )
        parser.add_argument(
            "--mera_entropy_coef",
            type=float,
            default=0.01,
            help="entropy regularization coefficient",
        )
        parser.add_argument(
            "--mera_actor_reg",
            type=float,
            default=0.0,
            help="coefficient on actor regularization term",
        )
        return parser

    def __init__(self, *input_args):
        args, env, actor, critic, buffer = input_args
        super().__init__(args, env, actor, buffer)

        self.critic_lr = args.critic_lr
        self.critic_decay = args.critic_decay
        self.tau = args.target_mitigate_coef
        self.mera_entropy_coef = args.mera_entropy_coef
        self.mera_actor_reg = args.mera_actor_reg

        self.critic = critic
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=args.critic_lr, weight_decay=args.critic_decay
        )
        self.do_actor_update = True
        self.do_critic_update = True

        self.registered_models.append((self.critic, self.critic_optimizer, "_critic"))

    def setup_monitors(self):
        super().setup_monitors()
        self.training_history.update(
            {
                "actor_loss": [],
                "critic_loss": [],
                "Q": [],
                "next_Q": [],
                "entropy": [],
            }
        )

    def step_train(self):
        observation, policy_output, user_feedback, done_mask, next_observation = self.buffer.sample(
            self.batch_size
        )
        reward = user_feedback["reward"].view(-1)

        current_critic_output = self.apply_critic(observation, policy_output, self.critic)
        current_q = current_critic_output["q"]

        next_policy_output = self.apply_policy(next_observation, self.actor_target, 0.0, False, True)
        target_critic_output = self.apply_critic(next_observation, next_policy_output, self.critic_target)
        next_q = target_critic_output["q"]
        target_q = reward + self.gamma * (done_mask * next_q).detach()

        critic_loss = F.mse_loss(current_q, target_q).mean()
        if self.do_critic_update and self.critic_lr > 0:
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        policy_output = self.apply_policy(
            observation, self.actor, 0.0, self.do_explore_in_train, True
        )
        critic_output = self.apply_critic(observation, policy_output, self.critic)
        entropy = policy_output["mera_entropy"].mean()
        actor_loss = -critic_output["q"].mean() - self.mera_entropy_coef * entropy
        if self.mera_actor_reg > 0:
            actor_loss = actor_loss + self.mera_actor_reg * policy_output["reg"]

        if self.do_actor_update and self.actor_lr > 0:
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        loss_dict = {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "Q": torch.mean(current_q).item(),
            "next_Q": torch.mean(next_q).item(),
            "entropy": entropy.item(),
        }

        for k, v in loss_dict.items():
            if k in self.training_history:
                self.training_history[k].append(v)
        return loss_dict

    def apply_policy(self, observation, actor, *policy_args):
        epsilon = policy_args[0]
        do_explore = policy_args[1]
        is_train = policy_args[2]
        input_dict = {
            "observation": observation,
            "candidates": self.env.get_candidate_info(observation),
            "epsilon": epsilon,
            "do_explore": do_explore,
            "is_train": is_train,
            "batch_wise": False,
        }
        return actor(input_dict)

    def apply_critic(self, observation, policy_output, critic):
        action = policy_output.get("mera_action", policy_output.get("hyper_action", policy_output["action"]))
        feed_dict = {"state": policy_output["state"], "action": action}
        return critic(feed_dict)

    def save(self):
        super().save()

    def load(self):
        super().load()
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)
