from src.environment.simulator import GroundTruthSimulator
from src.hybrid.single_agv_progress import (
    acknowledge_recovery_progress,
)


def make_case():
    simulator = GroundTruthSimulator(
        grid=[[0] * 4 for _ in range(2)],
        agent_positions={0: (0, 0), 1: (3, 1)},
        agent_goals={0: (3, 0), 1: (3, 1)},
    )

    plan = {
        "single_agv_completion": True,
        "active_agent": 0,
        "route_positions": [(0, 0), (1, 0)],
        "actions": {0: [3], 1: [0]},
    }

    return simulator, plan


def test_valid_recovery_progress():
    simulator, plan = make_case()
    before = dict(simulator.agent_positions)

    simulator.step_joint({0: 3, 1: 0})
    simulator.non_progress_steps = 9

    assert acknowledge_recovery_progress(
        simulator, plan, 0, before,
        {0: 3, 1: 0}, False,
    )
    assert simulator.non_progress_steps == 0


def test_wrong_action_does_not_reset_deadlock_counter():
    simulator, plan = make_case()
    before = dict(simulator.agent_positions)

    simulator.step_joint({0: 0, 1: 0})
    simulator.non_progress_steps = 9

    assert not acknowledge_recovery_progress(
        simulator, plan, 0, before,
        {0: 0, 1: 0}, False,
    )
    assert simulator.non_progress_steps == 9


def test_collision_does_not_reset_deadlock_counter():
    simulator, plan = make_case()
    before = dict(simulator.agent_positions)
    simulator.non_progress_steps = 9

    assert not acknowledge_recovery_progress(
        simulator, plan, 0, before,
        {0: 3, 1: 0}, True,
    )
    assert simulator.non_progress_steps == 9


def test_ordinary_plan_cannot_override_progress():
    simulator, plan = make_case()
    plan["single_agv_completion"] = False
    before = dict(simulator.agent_positions)

    simulator.step_joint({0: 3, 1: 0})
    simulator.non_progress_steps = 9

    assert not acknowledge_recovery_progress(
        simulator, plan, 0, before,
        {0: 3, 1: 0}, False,
    )
    assert simulator.non_progress_steps == 9
