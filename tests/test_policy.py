import numpy as np

from src.marl.networks import ActorNetwork
from src.marl.policy import select_action


def test_select_action_returns_valid_action():
    input_dim = 247

    actor = ActorNetwork(
        input_dim=input_dim,
        action_dim=5,
    )

    observation = np.zeros(
        input_dim,
        dtype=np.float32,
    )

    action, log_prob = select_action(
        actor,
        observation,
        deterministic=True,
    )

    assert int(action) in {0, 1, 2, 3, 4}
    assert isinstance(log_prob, float)
