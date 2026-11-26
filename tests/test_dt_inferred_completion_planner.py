"""Test bounded DT-inferred completion planning."""

from src.hybrid.dt_inferred_completion import (
    build_inferred_completion_plan,
)


def example():
    return {
        "grid": [[0] * 8],
        "positions": {
            0: (0, 0),
            1: (7, 0),
        },
        "goals": {
            0: (3, 0),
            1: (7, 0),
        },
        "remaining_steps": 10,
    }


def test_single_agv_completion():
    plan = build_inferred_completion_plan(**example())

    assert plan is not None
    assert plan["dt_inferred_completion"]
    assert plan["single_agv_completion"]
    assert plan["horizon"] == 3
    assert plan["active_agent"] == 0
    assert plan["route_positions"][-1] == (3, 0)


def test_two_agv_completion():
    args = example()
    args["goals"][1] = (5, 0)

    plan = build_inferred_completion_plan(**args)

    assert plan is not None
    assert plan["two_agv_completion"]
    assert plan["dt_inferred_completion"]
    assert plan["horizon"] == 5


def test_reject_insufficient_horizon():
    args = example()
    args["remaining_steps"] = 2

    assert build_inferred_completion_plan(**args) is None


def test_accept_exact_remaining_horizon():
    args = example()
    args["remaining_steps"] = 3

    assert build_inferred_completion_plan(**args) is not None


def test_reject_zero_remaining_steps():
    args = example()
    args["remaining_steps"] = 0

    assert build_inferred_completion_plan(**args) is None


def test_reject_blocked_route():
    args = example()
    args["grid"][0][2] = 1

    assert build_inferred_completion_plan(**args) is None


def test_reject_disabled_single_completion():
    args = example()
    args["enable_single"] = False

    assert build_inferred_completion_plan(**args) is None


def test_reject_disabled_two_completion():
    args = example()
    args["goals"][1] = (5, 0)
    args["enable_two"] = False

    assert build_inferred_completion_plan(**args) is None


def test_reject_all_agents_finished():
    args = example()
    args["goals"] = dict(args["positions"])

    assert build_inferred_completion_plan(**args) is None


def test_reject_missing_position():
    args = example()
    del args["positions"][1]

    assert build_inferred_completion_plan(**args) is None


def test_expected_positions_cover_entire_route():
    plan = build_inferred_completion_plan(**example())

    expected = plan["recovery_expected_positions"]

    assert len(expected) == plan["horizon"] + 1
    assert expected[0] == example()["positions"]
    assert expected[-1] == example()["goals"]


def test_finished_agent_remains_stationary():
    plan = build_inferred_completion_plan(**example())

    assert all(
        state[1] == (7, 0)
        for state in plan["recovery_expected_positions"]
    )


def test_only_one_agent_moves_per_step():
    args = example()
    args["goals"][1] = (5, 0)

    plan = build_inferred_completion_plan(**args)
    expected = plan["recovery_expected_positions"]

    for before, after in zip(expected, expected[1:]):
        assert sum(
            before[a] != after[a] for a in before
        ) == 1
