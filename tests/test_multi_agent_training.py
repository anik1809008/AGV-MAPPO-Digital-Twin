import numpy as np

from src.marl.multi_agent_buffer import MultiAgentRolloutBuffer
from src.marl.multi_agent_training import flatten_multi_agent_rollout


def test_flatten_multi_agent_rollout():
    buffer = MultiAgentRolloutBuffer(
        num_agents=8,
    )

    observations = [
        np.zeros(247, dtype=np.float32)
        for _ in range(8)
    ]

    for _ in range(2):
        buffer.add_step(
            agent_observations=observations,
            actions=[0, 1, 2, 3, 4, 0, 1, 2],
            log_probs=[-1.0] * 8,
            rewards=[0.1] * 8,
            value=0.5,
            dones=[False] * 8,
            centralized_state=np.zeros(
                1976,
                dtype=np.float32,
            ),
        )

    data = flatten_multi_agent_rollout(
        buffer
    )

    assert data["observations"].shape == (
        16,
        247,
    )

    assert data["actions"].shape == (
        16,
    )

    assert data["centralized_states"].shape == (
        16,
        1976,
    )
