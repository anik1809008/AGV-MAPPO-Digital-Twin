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
