import numpy as np

from src.marl.episode_training import train_from_episode
from src.marl.mappo_trainer import MAPPOTrainer
from src.marl.multi_agent_buffer import MultiAgentRolloutBuffer
from src.marl.networks import ActorNetwork, CriticNetwork


def test_train_from_episode_clears_buffer():
    buffer = MultiAgentRolloutBuffer(
        num_agents=2,
    )

    observations = [
        np.zeros(247, dtype=np.float32)
        for _ in range(2)
    ]

    for t in range(4):
        buffer.add_step(
            agent_observations=observations,
            actions=[0, 1],
            log_probs=[-1.0, -1.0],
            rewards=[1.0, 1.0],
            value=0.5,
            dones=[t == 3, t == 3],
            centralized_state=np.zeros(
                494,
                dtype=np.float32,
            ),
        )

    actor = ActorNetwork(
        input_dim=247,
        action_dim=5,
    )

    critic = CriticNetwork(
        input_dim=494,
    )

    trainer = MAPPOTrainer(
        actor=actor,
        critic=critic,
    )

    history = train_from_episode(
        trainer=trainer,
        multi_agent_buffer=buffer,
        epochs=2,
        minibatch_size=4,
    )

    assert len(history) == 4
    assert len(buffer) == 0
