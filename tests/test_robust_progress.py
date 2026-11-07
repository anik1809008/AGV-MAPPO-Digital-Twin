from src.environment.actions import Action
from src.safety.shield import SafetyShield
from src.safety.robust_progress import select_robust_progress_action


GRID = [[0] * 8 for _ in range(8)]


def select(positions, goal, occupancy=None):
    return select_robust_progress_action(
        shield=SafetyShield(GRID),
        agent_id=0,
        ranked_actions=[
            (Action.WAIT, 0.8),
            (Action.EAST, 0.2),
        ],
        possible_current_positions=positions,
        goal_position=goal,
        other_current_positions={},
        other_next_positions={},
        reachable_occupancies=occupancy or {},
        other_possible_transitions={},
    )


def test_selects_progress_over_wait():
    result = select(
        positions={(1, 1), (2, 1)},
        goal=(6, 1),
    )
    assert result[0] == Action.EAST


def test_rejects_unsafe_progress():
    result = select(
        positions={(1, 1), (2, 1)},
        goal=(6, 1),
        occupancy={1: {(3, 1)}},
    )
    assert result is None


def test_rejects_action_that_fails_for_one_position():
    result = select(
        positions={(6, 1), (5, 1)},
        goal=(6, 1),
    )
    assert result is None


def test_rejects_empty_position_set():
    try:
        select(positions=set(), goal=(6, 1))
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")
