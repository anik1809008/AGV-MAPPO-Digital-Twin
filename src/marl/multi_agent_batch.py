import numpy as np
import torch

from src.marl.multi_agent_training import flatten_multi_agent_rollout
from src.marl.multi_agent_gae import compute_multi_agent_gae


def prepare_multi_agent_batch(
    multi_agent_buffer,
    next_values=None,
    gamma=0.99,
    gae_lambda=0.95,
):
    data = flatten_multi_agent_rollout(
        multi_agent_buffer
    )

    advantages, returns = compute_multi_agent_gae(
        multi_agent_buffer=multi_agent_buffer,
        next_values=next_values,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )

    advantages = advantages.T.reshape(-1)
    returns = returns.T.reshape(-1)

    advantages = torch.tensor(
        advantages,
        dtype=torch.float32,
    )

    if len(advantages) > 1:
        advantages = (
            advantages - advantages.mean()
        ) / (
            advantages.std(unbiased=False) + 1e-8
        )

    return {
        "observations": torch.tensor(
            data["observations"],
            dtype=torch.float32,
        ),
        "centralized_states": torch.tensor(
            data["centralized_states"],
            dtype=torch.float32,
        ),
        "actions": torch.tensor(
            data["actions"],
            dtype=torch.long,
        ),
        "old_log_probs": torch.tensor(
            data["log_probs"],
            dtype=torch.float32,
        ),
        "advantages": advantages,
        "returns": torch.tensor(
            returns,
            dtype=torch.float32,
        ),
    }
