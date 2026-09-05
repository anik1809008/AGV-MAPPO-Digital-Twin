import numpy as np

from src.marl.buffer import RolloutBuffer
from src.marl.networks import ActorNetwork, CriticNetwork
from src.marl.mappo_trainer import MAPPOTrainer
from src.marl.training_pipeline import train_from_buffer


def test_train_from_buffer_runs_multi_epoch_updates():
    buffer = RolloutBuffer()

    for i in range(8):
        buffer.add(
            observation=np.zeros(247, dtype=np.float32),
            action=i % 5,
            log_prob=-1.0,
            reward=1.0,
            value=0.5,
            done=(i == 7),
        )

    centralized_states = [
        np.zeros(1976, dtype=np.float32)
        for _ in range(8)
    ]

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

    history = train_from_buffer(
        trainer=trainer,
        buffer=buffer,
        centralized_states=centralized_states,
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
