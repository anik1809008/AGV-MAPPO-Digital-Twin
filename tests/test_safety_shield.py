from src.environment.actions import Action
from src.safety.shield import SafetyShield


def test_action_can_be_safe_when_blocked_from_one_possible_position():
    grid = [
        [0, 0, 1],
        [0, 0, 0],
    ]

    shield = SafetyShield(grid)

    safe = shield.is_action_safe(
        agent_id=0,
        possible_current_positions={
            (1, 0),
            (1, 1),
        },
        action=Action.EAST,
        other_current_positions={},
        other_next_positions={},
        reachable_occupancies={
            0: {
                (1, 0),
                (1, 1),
            },
        },
        other_possible_transitions={},
    )

    assert safe is True
