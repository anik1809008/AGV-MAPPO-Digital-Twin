import numpy as np

from src.marl.multi_agent_buffer import MultiAgentRolloutBuffer
from src.marl.multi_agent_batch import prepare_multi_agent_batch


def test_prepare_multi_agent_batch():
    buffer = MultiAgentRolloutBuffer(
        num_agents=2,
    )

    observations = [
        np.zeros(247, dtype=np.float32)
        for _ in range(2)
    ]

    for t in range(2):
        buffer.add_step(
            agent_observations=observations,
            actions=[0, 1],
            log_probs=[-1.0, -1.0],
            rewards=[1.0, 1.0],
            value=0.5,
            dones=[t == 1, t == 1],
            centralized_state=np.zeros(
                494,
                dtype=np.float32,
            ),
        )

    batch = prepare_multi_agent_batch(
        buffer
    )

    assert batch["observations"].shape == (
        4,
        247,
    )

    assert batch["actions"].shape == (
        4,
    )

    assert batch["centralized_states"].shape == (
        4,
        494,
    )

    assert batch["advantages"].shape == (
        4,
    )

    assert batch["returns"].shape == (
        4,
    )
