import numpy as np

from src.marl.multi_agent_buffer import MultiAgentRolloutBuffer


def test_multi_agent_rollout_buffer():
    buffer = MultiAgentRolloutBuffer(
        num_agents=8,
    )

    observations = [
        np.zeros(247, dtype=np.float32)
        for _ in range(8)
    ]

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

    assert len(buffer) == 1
    assert len(buffer.agent_buffers) == 8
    assert len(buffer.agent_buffers[0]) == 1
    assert len(buffer.centralized_states) == 1

    buffer.clear()

    assert len(buffer) == 0
