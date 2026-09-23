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
from src.environment.actions import Action
class DummyShield:
    def is_action_safe(
        self,
        agent_id,
        possible_current_positions,
        action,
        other_current_positions,
        other_next_positions,
        reachable_occupancies,
        other_possible_transitions=None,
    ):
        return action in {
            Action.WAIT,
            Action.SOUTH,
        }


def test_filter_prefers_safe_movement_over_wait_after_intervention():
    filter_ = ShieldActionFilter()
    shield = DummyShield()

    ranked_actions = [
        (Action.NORTH, 0.80),
        (Action.WAIT, 0.10),
        (Action.SOUTH, 0.05),
        (Action.EAST, 0.03),
        (Action.WEST, 0.02),
    ]

    action, probability = filter_.select_safe_action(
        shield=shield,
        agent_id=0,
        ranked_actions=ranked_actions,
        possible_current_positions={(1, 1)},
        other_current_positions={},
        other_next_positions={},
        reachable_occupancies={},
        other_possible_transitions={},
    )

    assert action == Action.SOUTH
    assert probability == 0.05
