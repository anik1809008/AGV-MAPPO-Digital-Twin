import numpy as np

from src.marl.buffer import RolloutBuffer
from src.marl.networks import ActorNetwork, CriticNetwork
from src.marl.mappo_trainer import MAPPOTrainer
from src.marl.training_pipeline import train_from_buffer


def test_train_from_buffer_returns_metrics():
    buffer = RolloutBuffer()

    for i in range(4):
        buffer.add(
            observation=np.zeros(247, dtype=np.float32),
            action=i,
            log_prob=-1.0,
            reward=1.0,
            value=0.5,
            done=(i == 3),
        )

    centralized_states = [
        np.zeros(1976, dtype=np.float32)
        for _ in range(4)
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

    metrics = train_from_buffer(
        trainer=trainer,
        buffer=buffer,
        centralized_states=centralized_states,
    )

    assert set(metrics.keys()) == {
        "total_loss",
        "actor_loss",
        "critic_loss",
        "entropy",
    }
