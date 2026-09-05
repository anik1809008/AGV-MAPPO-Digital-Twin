import numpy as np
import torch

from src.marl.gae import compute_gae


def prepare_training_batch(
    buffer,
    centralized_states,
    next_value=0.0,
    gamma=0.99,
    gae_lambda=0.95,
):
    advantages, returns = compute_gae(
        rewards=buffer.rewards,
        values=buffer.values,
        dones=buffer.dones,
        next_value=next_value,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )

    observations = torch.tensor(
        np.asarray(buffer.observations),
        dtype=torch.float32,
    )

    centralized_states = torch.tensor(
        np.asarray(centralized_states),
        dtype=torch.float32,
    )

    actions = torch.tensor(
        buffer.actions,
        dtype=torch.long,
    )

    old_log_probs = torch.tensor(
        buffer.log_probs,
        dtype=torch.float32,
    )

    advantages = torch.tensor(
        advantages,
        dtype=torch.float32,
    )

    returns = torch.tensor(
        returns,
        dtype=torch.float32,
    )

    if len(advantages) > 1:
        advantages = (
            advantages - advantages.mean()
        ) / (
            advantages.std(unbiased=False) + 1e-8
        )

    return {
        "observations": observations,
        "centralized_states": centralized_states,
        "actions": actions,
        "old_log_probs": old_log_probs,
        "advantages": advantages,
        "returns": returns,
    }


def train_from_buffer(
    trainer,
    buffer,
    centralized_states,
    next_value=0.0,
    gamma=0.99,
    gae_lambda=0.95,
    epochs=4,
    minibatch_size=64,
):
    batch = prepare_training_batch(
        buffer=buffer,
        centralized_states=centralized_states,
        next_value=next_value,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )

    return trainer.update_epochs(
        **batch,
        epochs=epochs,
        minibatch_size=minibatch_size,
    )
