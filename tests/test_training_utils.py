import numpy as np

from src.marl.buffer import RolloutBuffer
from src.marl.gae import compute_gae


def test_rollout_buffer_add_and_clear():
    buffer = RolloutBuffer()

    buffer.add(
        observation=np.zeros(247, dtype=np.float32),
        action=2,
        log_prob=-0.5,
        reward=0.1,
        value=0.25,
        done=False,
    )

    assert len(buffer) == 1
    assert buffer.actions[0] == 2
    assert buffer.rewards[0] == 0.1

    buffer.clear()

    assert len(buffer) == 0


def test_compute_gae():
    advantages, returns = compute_gae(
        rewards=[1.0, 1.0],
        values=[0.5, 0.5],
        dones=[False, True],
        next_value=0.0,
    )

    assert np.allclose(
        advantages,
        [1.46525, 0.5],
    )

    assert np.allclose(
        returns,
        [1.96525, 1.0],
    )
