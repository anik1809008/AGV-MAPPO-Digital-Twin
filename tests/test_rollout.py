import numpy as np

from src.marl.networks import ActorNetwork, CriticNetwork
from src.marl.rollout import collect_single_step


def test_collect_single_step_for_eight_agents():
    actor = ActorNetwork(
        input_dim=247,
        action_dim=5,
    )

    critic = CriticNetwork(
        input_dim=1976,
    )

    observations = [
        np.zeros(247, dtype=np.float32)
        for _ in range(8)
    ]

    result = collect_single_step(
        actor=actor,
        critic=critic,
        agent_observations=observations,
    )

    assert len(result["actions"]) == 8
    assert len(result["log_probs"]) == 8
    assert result["centralized_state"].shape == (1976,)
    assert isinstance(result["value"], float)
from src.marl.multi_agent_buffer import MultiAgentRolloutBuffer
from src.marl.rollout import collect_and_store_step


def test_collect_and_store_step():
    actor = ActorNetwork(
        input_dim=247,
        action_dim=5,
    )

    critic = CriticNetwork(
        input_dim=1976,
    )

    buffer = MultiAgentRolloutBuffer(
        num_agents=8,
    )

    observations = [
        np.zeros(247, dtype=np.float32)
        for _ in range(8)
    ]

    result = collect_and_store_step(
        actor=actor,
        critic=critic,
        agent_observations=observations,
        rewards=[0.1] * 8,
        dones=[False] * 8,
        multi_agent_buffer=buffer,
    )

    assert len(result["actions"]) == 8
    assert len(buffer) == 1
    assert len(buffer.agent_buffers[0]) == 1
