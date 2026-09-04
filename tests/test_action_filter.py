from src.environment.actions import Action
from src.safety.action_filter import ShieldActionFilter
from src.safety.shield import SafetyShield


def test_selects_next_highest_safe_action_and_counts_intervention():
    shield = SafetyShield([
        [0, 0, 0],
        [0, 0, 0],
    ])

    filter_ = ShieldActionFilter()

    ranked_actions = [
        (Action.EAST, 0.50),
        (Action.SOUTH, 0.30),
        (Action.WAIT, 0.20),
    ]

    action, probability = filter_.select_safe_action(
        shield=shield,
        agent_id=0,
        ranked_actions=ranked_actions,
        possible_current_positions={(0, 0)},
        other_current_positions={1: (2, 0)},
        other_next_positions={1: (2, 0)},
        reachable_occupancies={
            1: {(1, 0), (2, 0)},
        },
    )

    assert action == Action.SOUTH
    assert probability == 0.30
    assert filter_.intervention_count == 1
