from unittest.mock import patch

from src.mapf.lns_sat_uncertain_runner import run_lns_sat_uncertain_episode


GRID = [[0] * 5]


def test_success_without_execution_delay():
    solution = {
        "paths": {0: [(0, 0), (1, 0), (2, 0)]},
        "makespan": 2,
    }

    with patch(
        "src.mapf.lns_sat_uncertain_runner.solve_lns_sat",
        return_value=solution,
    ):
        result = run_lns_sat_uncertain_episode(
            GRID, {0: (0, 0)}, {0: (2, 0)},
            immediate_probability=1.0,
        )

    assert result["success"]
    assert result["makespan"] == 2
    assert result["path_length"] == 2
    assert not result["collision"]


def test_initial_planning_failure():
    with patch(
        "src.mapf.lns_sat_uncertain_runner.solve_lns_sat",
        return_value=None,
    ):
        result = run_lns_sat_uncertain_episode(
            GRID, {0: (0, 0)}, {0: (2, 0)},
        )

    assert result["planning_failed"]
    assert not result["success"]
    assert result["makespan"] == 0


def test_all_delayed_commands_can_still_finish():
    solution = {
        "paths": {0: [(0, 0), (1, 0), (2, 0)]},
        "makespan": 2,
    }

    with patch(
        "src.mapf.lns_sat_uncertain_runner.solve_lns_sat",
        return_value=solution,
    ):
        result = run_lns_sat_uncertain_episode(
            GRID, {0: (0, 0)}, {0: (2, 0)},
            immediate_probability=0.0,
            max_steps=10,
        )

    assert result["success"]
    assert result["path_length"] == 2
    assert result["makespan"] >= 3


def test_delayed_execution_detects_multi_agent_collision():
    """Delayed departure causes a vertex collision."""
    solution = {
        "paths": {
            0: [(0, 0), (1, 0), (2, 0)],
            1: [(1, 1), (1, 1), (1, 0)],
        },
        "makespan": 2,
    }

    with patch(
        "src.mapf.lns_sat_uncertain_runner.solve_lns_sat",
        return_value=solution,
    ), patch(
        "src.mapf.lns_sat_uncertain_runner.ExecutionDelayModel.sample_delay",
        side_effect=[0, 0, 1, 0],
    ):
        result = run_lns_sat_uncertain_episode(
            [[0] * 5 for _ in range(2)],
            {0: (0, 0), 1: (1, 1)},
            {0: (2, 0), 1: (1, 0)},
            immediate_probability=0.5,
            max_steps=10,
        )

    assert result["collision"] is True
    assert result["success"] is False
    assert result["makespan"] == 2
