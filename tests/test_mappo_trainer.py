import torch

from src.marl.networks import ActorNetwork, CriticNetwork
from src.marl.mappo_trainer import MAPPOTrainer


def test_mappo_trainer_update_returns_finite_metrics():
    actor = ActorNetwork(
        input_dim=247,
        action_dim=5,
    )

    critic = CriticNetwork(
        input_dim=1976,
    )

    trainer = MAPPOTrainer(
        actor=actor,
        critic=critic,
    )

    batch_size = 4

    metrics = trainer.update(
        observations=torch.zeros(batch_size, 247),
        centralized_states=torch.zeros(batch_size, 1976),
        actions=torch.tensor([0, 1, 2, 3], dtype=torch.long),
        old_log_probs=torch.zeros(batch_size),
        advantages=torch.tensor([1.0, 0.5, -0.2, 0.3]),
        returns=torch.tensor([1.0, 0.8, 0.2, 0.5]),
    )

    assert set(metrics.keys()) == {
        "total_loss",
        "actor_loss",
        "critic_loss",
        "entropy",
    }

    for value in metrics.values():
        assert torch.isfinite(torch.tensor(value))
def test_mappo_trainer_multi_epoch_updates():
    actor = ActorNetwork(
        input_dim=247,
        action_dim=5,
    )

    critic = CriticNetwork(
        input_dim=1976,
    )

    trainer = MAPPOTrainer(
        actor=actor,
        critic=critic,
    )

    batch_size = 8

    history = trainer.update_epochs(
        observations=torch.zeros(batch_size, 247),
        centralized_states=torch.zeros(batch_size, 1976),
        actions=torch.tensor(
            [0, 1, 2, 3, 4, 0, 1, 2],
            dtype=torch.long,
        ),
        old_log_probs=torch.zeros(batch_size),
        advantages=torch.ones(batch_size),
        returns=torch.ones(batch_size),
        epochs=2,
        minibatch_size=4,
    )

    assert len(history) == 4

    for metrics in history:
        assert set(metrics.keys()) == {
            "total_loss",
            "actor_loss",
            "critic_loss",
            "entropy",
        }
