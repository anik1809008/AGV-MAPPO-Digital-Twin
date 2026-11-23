"""Regression tests for route-aware two-AGV progress."""

from src.environment.simulator import GroundTruthSimulator
from src.hybrid.recovery_route_guard import build_expected_positions
from src.hybrid.two_agv_progress import acknowledge_two_agv_progress


def make_case():
    starts = {0: (2, 0), 1: (5, 0)}
    goals = {0: (0, 0), 1: (4, 0)}
    actions = {0: [3], 1: [0]}

    simulator = GroundTruthSimulator(
        [[0] * 7], starts, goals
    )

    plan = {
        "two_agv_completion": True,
        "actions": actions,
        "recovery_expected_positions": build_expected_positions(
            starts, actions, 1
        ),
    }

    return simulator, plan


def test_verified_detour_resets_non_progress():
    simulator, plan = make_case()
    before = dict(simulator.agent_positions)
    simulator.non_progress_steps = 5

    result = simulator.step_joint({0: 3, 1: 0})
    assert not result["collision"]

    assert acknowledge_two_agv_progress(
        simulator, plan, 0, before, {0: 3, 1: 0}, False
    )
    assert simulator.non_progress_steps == 0


def test_blocked_movement_cannot_reset_progress():
    simulator, plan = make_case()
    before = dict(simulator.agent_positions)
    simulator.non_progress_steps = 5

    simulator.step_joint({0: 0, 1: 0})

    assert not acknowledge_two_agv_progress(
        simulator, plan, 0, before, {0: 3, 1: 0}, False
    )
    assert simulator.non_progress_steps >= 5


def test_collision_cannot_reset_progress():
    simulator, plan = make_case()
    before = dict(simulator.agent_positions)
    simulator.non_progress_steps = 5

    simulator.step_joint({0: 3, 1: 0})

    assert not acknowledge_two_agv_progress(
        simulator, plan, 0, before, {0: 3, 1: 0}, True
    )
    assert simulator.non_progress_steps >= 5
