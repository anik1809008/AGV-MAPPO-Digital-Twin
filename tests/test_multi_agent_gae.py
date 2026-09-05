import numpy as np

from src.marl.multi_agent_buffer import MultiAgentRolloutBuffer
from src.marl.multi_agent_gae import compute_multi_agent_gae


def test_compute_multi_agent_gae():
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
            dones=[
                t == 1,
                t == 1,
            ],
            centralized_state=np.zeros(
                494,
                dtype=np.float32,
            ),
        )

    advantages, returns = compute_multi_agent_gae(
        buffer
    )

    assert advantages.shape == (2, 2)
    assert returns.shape == (2, 2)

    assert np.allclose(
        advantages[0],
        [1.46525, 0.5],
    )

    assert np.allclose(
        returns[0],
        [1.96525, 1.0],
    )
