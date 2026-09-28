from src.environment.actions import Action
from src.safety.m7_action_filter import M7ActionFilter
from src.safety.shield import SafetyShield


def test_m7_prefers_safe_bfs_progress_when_stagnating():
    grid = [
        [0, 0, 0],
        [0, 0, 0],
    ]

    shield = SafetyShield(grid)
    action_filter = M7ActionFilter()

    action, _ = action_filter.select_safe_action(
        shield=shield,
        agent_id=0,
        ranked_actions=[
            (Action.SOUTH, 0.9),
            (Action.EAST, 0.6),
            (Action.WAIT, 0.1),
        ],
        possible_current_positions={(0, 0)},
        other_current_positions={},
        other_next_positions={},
        reachable_occupancies={
            0: {(0, 0)},
        },
        current_position=(0, 0),
        goal_position=(2, 0),
        prefer_progress=True,
    )

    assert action == Action.EAST
