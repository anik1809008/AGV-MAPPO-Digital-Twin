from src.environment.actions import Action
from src.safety.shield import SafetyShield


def test_reserved_next_position_causes_vertex_conflict():
    shield = SafetyShield([
        [0, 0, 0],
    ])

    reserved_next_positions = {
        0: (1, 0),
    }

    safe = shield.is_action_safe(
        agent_id=1,
        possible_current_positions={(2, 0)},
        action=Action.WEST,
        other_current_positions={
            0: (0, 0),
        },
        other_next_positions=reserved_next_positions,
        reachable_occupancies={
            0: {(0, 0)},
            1: {(2, 0)},
        },
    )

    assert safe is False
