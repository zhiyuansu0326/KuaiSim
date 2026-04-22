import copy
import torch
import torch.nn.functional as F

from model.agent.BaseRLAgent import BaseRLAgent


class MERAComplete(BaseRLAgent):
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
            "--merac_actor_reg",
            type=float,
            default=0.0,
            help="coefficient on actor regularization term",
        )
        parser.add_argument(
            "--merac_use_legacy_done_mask",
            type=int,
            default=1,
            help="1: use done_mask * next_q (legacy); 0: use (1-done_mask) * next_q",
        )
        return parser

    def __init__(self, *input_args):
        args, env, actor, critic, buffer = input_args
        super().__init__(args, env, actor, buffer)

        if not hasattr(args, "merac_tau"):
            raise ValueError("MERAComplete requires --merac_tau from MERACompletePolicy.")

        self.critic_lr = args.critic_lr
        self.critic_decay = args.critic_decay
        self.soft_update_tau = args.target_mitigate_coef
        self.merac_tau = args.merac_tau
        self.merac_actor_reg = args.merac_actor_reg
        self.use_legacy_done_mask = bool(args.merac_use_legacy_done_mask)

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
                "actor_loss_kkt": [],
                "critic_loss": [],
                "Q": [],
                "next_Q": [],
                "entropy": [],
            }
        )

    def _project_matrix_shift_invariant(self, matrix):
        row_mean = matrix.mean(dim=2, keepdim=True)
        col_mean = matrix.mean(dim=1, keepdim=True)
        global_mean = matrix.mean(dim=(1, 2), keepdim=True)
        return matrix - row_mean - col_mean + global_mean

    def step_train(self):
        observation, policy_output, user_feedback, done_mask, next_observation = self.buffer.sample(
            self.batch_size
        )
        reward = user_feedback["reward"].view(-1)

        current_critic_output = self.apply_critic(policy_output, self.critic)
        current_q = current_critic_output["q"]

        next_policy_output = self.apply_policy(next_observation, self.actor_target, 0.0, False, True)
        target_critic_output = self.apply_critic(next_policy_output, self.critic_target)
        next_q = target_critic_output["q"]
        next_entropy = next_policy_output["mera_entropy"].view(-1)

        if self.use_legacy_done_mask:
            discount_mask = done_mask.to(next_q.dtype)
        else:
            discount_mask = (1 - done_mask.to(next_q.dtype))
        target_q = reward + self.gamma * (discount_mask * (next_q + self.merac_tau * next_entropy)).detach()

        critic_loss = F.mse_loss(current_q, target_q).mean()
        if self.do_critic_update and self.critic_lr > 0:
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        policy_output = self.apply_policy(
            observation, self.actor, 0.0, self.do_explore_in_train, True
        )
        soft_action = policy_output["soft_action_matrix"]
        hard_action = policy_output["hard_action_matrix"].detach()
        action_st = hard_action - soft_action.detach() + soft_action

        critic_output_for_grad = self.apply_critic(
            policy_output, self.critic, action_matrix_override=action_st
        )
        grad_action = torch.autograd.grad(
            critic_output_for_grad["q"].sum(),
            action_st,
            retain_graph=True,
            create_graph=False,
        )[0]
        g_theta = (-grad_action).detach()

        residual = self._project_matrix_shift_invariant(
            policy_output["logits"] + policy_output["gumbel_noise"] - g_theta
        )
        actor_loss_kkt = 0.5 * residual.pow(2).mean()
        actor_loss = actor_loss_kkt
        if self.merac_actor_reg > 0:
            actor_loss = actor_loss + self.merac_actor_reg * policy_output["reg"]

        if self.do_actor_update and self.actor_lr > 0:
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(
                self.soft_update_tau * param.data + (1 - self.soft_update_tau) * target_param.data
            )
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(
                self.soft_update_tau * param.data + (1 - self.soft_update_tau) * target_param.data
            )

        entropy = policy_output["mera_entropy"].mean()
        loss_dict = {
            "actor_loss": actor_loss.item(),
            "actor_loss_kkt": actor_loss_kkt.item(),
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

    def apply_critic(self, policy_output, critic, action_matrix_override=None):
        action_matrix = action_matrix_override
        if action_matrix is None:
            action_matrix = policy_output.get("hard_action_matrix")
            if action_matrix is None:
                hard_action = policy_output.get("merac_hard_action", policy_output["action"])
                action_matrix = hard_action.reshape(
                    -1, self.actor.shortlist_size, self.actor.matrix_cols
                )
        feed_dict = {
            "state": policy_output["state"],
            "action_matrix": action_matrix,
            "shortlist_indices": policy_output["shortlist_indices"],
        }
        return critic(feed_dict)

    def save(self):
        super().save()

    def load(self):
        super().load()
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)
