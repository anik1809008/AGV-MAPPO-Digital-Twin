from src.environment.actions import Action
from src.safety.local_recovery import LocalRecoveryPlanner
def test_local_recovery_moves_toward_goal_when_free():
    planner = LocalRecoveryPlanner([
        [0, 0, 0],
        [0, 0, 0],
    ])

    action = planner.select_recovery_action(
        agent_id=0,
        current_position=(0, 0),
        goal_position=(2, 0),
        occupied_positions={(2, 1)},
    )

    assert action == Action.EAST

def test_local_recovery_uses_detour_when_next_step_is_blocked():
    planner = LocalRecoveryPlanner([
        [0, 0, 0],
        [0, 0, 0],
    ])

    action = planner.select_recovery_action(
        agent_id=0,
        current_position=(0, 0),
        goal_position=(2, 0),
        occupied_positions={(1, 0)},
    )

    assert action == Action.SOUTH
def test_local_recovery_waits_when_all_moves_are_blocked():
    planner = LocalRecoveryPlanner([
        [0, 0],
        [0, 0],
    ])

    action = planner.select_recovery_action(
        agent_id=0,
        current_position=(0, 0),
        goal_position=(1, 0),
        occupied_positions={
            (1, 0),
            (0, 1),
        },
    )

    assert action == Action.WAIT
