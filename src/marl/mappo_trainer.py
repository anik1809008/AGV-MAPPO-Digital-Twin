import torch

from src.marl.mappo_update import compute_mappo_losses


class MAPPOTrainer:
    def __init__(
        self,
        actor,
        critic,
        actor_lr=3e-4,
        critic_lr=1e-3,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
    ):
        self.actor = actor
        self.critic = critic

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=actor_lr,
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=critic_lr,
        )

        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

    def update(
        self,
        observations,
        centralized_states,
        actions,
        old_log_probs,
        advantages,
        returns,
    ):
        losses = compute_mappo_losses(
            actor=self.actor,
            critic=self.critic,
            observations=observations,
            centralized_states=centralized_states,
            actions=actions,
            old_log_probs=old_log_probs,
            advantages=advantages,
            returns=returns,
            clip_epsilon=self.clip_epsilon,
            entropy_coef=self.entropy_coef,
            value_coef=self.value_coef,
        )

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        losses["total_loss"].backward()

        torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            self.max_grad_norm,
        )

        torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            self.max_grad_norm,
        )

        self.actor_optimizer.step()
        self.critic_optimizer.step()

        return {
            "total_loss": float(losses["total_loss"].detach()),
            "actor_loss": float(losses["actor_loss"].detach()),
            "critic_loss": float(losses["critic_loss"].detach()),
            "entropy": float(losses["entropy"].detach()),
        }
