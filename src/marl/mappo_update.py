import torch
import torch.nn.functional as F
from torch.distributions import Categorical


def compute_mappo_losses(
    actor,
    critic,
    observations,
    centralized_states,
    actions,
    old_log_probs,
    advantages,
    returns,
    clip_epsilon=0.2,
    entropy_coef=0.01,
    value_coef=0.5,
):
    logits = actor(observations)
    distribution = Categorical(logits=logits)

    new_log_probs = distribution.log_prob(actions)
    entropy = distribution.entropy().mean()

    ratios = torch.exp(
        new_log_probs - old_log_probs
    )

    surrogate_1 = ratios * advantages
    surrogate_2 = torch.clamp(
        ratios,
        1.0 - clip_epsilon,
        1.0 + clip_epsilon,
    ) * advantages

    actor_loss = -torch.min(
        surrogate_1,
        surrogate_2,
    ).mean()

    values = critic(
        centralized_states
    ).squeeze(-1)

    critic_loss = F.mse_loss(
        values,
        returns,
    )

    total_loss = (
        actor_loss
        + value_coef * critic_loss
        - entropy_coef * entropy
    )

    return {
        "total_loss": total_loss,
        "actor_loss": actor_loss,
        "critic_loss": critic_loss,
        "entropy": entropy,
    }
