import numpy as np

from src.marl.networks import ActorNetwork
from src.safety.m4_controller import M4Controller
from src.safety.shield import SafetyShield


def test_m4_controller_returns_valid_action():
    actor = ActorNetwork(
        input_dim=247,
        action_dim=5,
    )

    shield = SafetyShield([
        [0, 0, 0],
        [0, 0, 0],
    ])

    controller = M4Controller(
        actor=actor,
        shield=shield,
    )

    observation = np.zeros(
        247,
        dtype=np.float32,
    )

    action, probability = controller.select_action(
        observation_vector=observation,
        agent_id=0,
        possible_current_positions={(0, 0)},
        other_current_positions={1: (2, 0)},
        other_next_positions={1: (2, 0)},
        reachable_occupancies={
            1: {(1, 0), (2, 0)},
        },
    )

    assert int(action) in {0, 1, 2, 3, 4}
    assert isinstance(probability, float)
    assert controller.action_filter.intervention_count in {0, 1}
