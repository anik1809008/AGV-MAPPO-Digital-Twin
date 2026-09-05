import torch

from src.marl.networks import ActorNetwork, CriticNetwork
from src.marl.mappo_update import compute_mappo_losses


def test_compute_mappo_losses():
    actor = ActorNetwork(
        input_dim=247,
        action_dim=5,
    )

    critic = CriticNetwork(
        input_dim=1976,
    )

    batch_size = 4

    observations = torch.zeros(
        batch_size,
        247,
    )

    centralized_states = torch.zeros(
        batch_size,
        1976,
    )

    actions = torch.tensor(
        [0, 1, 2, 3],
        dtype=torch.long,
    )

    old_log_probs = torch.zeros(
        batch_size,
    )

    advantages = torch.tensor(
        [1.0, 0.5, -0.2, 0.3],
    )

    returns = torch.tensor(
        [1.0, 0.8, 0.2, 0.5],
    )

    losses = compute_mappo_losses(
        actor=actor,
        critic=critic,
        observations=observations,
        centralized_states=centralized_states,
        actions=actions,
        old_log_probs=old_log_probs,
        advantages=advantages,
        returns=returns,
    )

    assert torch.isfinite(losses["total_loss"])
    assert torch.isfinite(losses["actor_loss"])
    assert torch.isfinite(losses["critic_loss"])
    assert losses["entropy"] > 0
