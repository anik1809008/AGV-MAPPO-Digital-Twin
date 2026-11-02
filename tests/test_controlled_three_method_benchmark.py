from types import SimpleNamespace
from unittest.mock import patch

from scripts.run_controlled_three_method_benchmark import run_trial


INSTANCE = {
    "grid": [[0, 0, 0]],
    "starts": {0: (0, 0)},
    "goals": {0: (2, 0)},
}

ARGS = SimpleNamespace(
    seed=1,
    immediate_probability=1.0,
    max_steps=10,
    aoi_threshold=2,
)


def test_classical_dispatch_and_timing():
    classical = {
        "success": True,
        "collision": False,
        "deadlock": False,
        "planning_failed": False,
        "makespan": 2,
        "path_length": 2,
        "planning_time": 0.01,
    }

    with patch(
        "scripts.run_controlled_three_method_benchmark."
        "run_lns_sat_uncertain_episode",
        return_value=classical,
    ) as runner:
        result = run_trial("LNS-SAT", INSTANCE, ARGS, {})

    runner.assert_called_once()
    assert result["success"] is True
    assert result["lns_planning_time"] == 0.01
    assert result["episode_wall_time"] >= 0
    assert result["planning_failures"] == 0


def test_m7_dispatch_and_metrics():
    metrics = {
        "success": True,
        "collision": False,
        "deadlock": False,
        "makespan": 5,
        "path_length": 8,
        "planning_time": 0.2,
    }

    with patch(
        "scripts.run_controlled_three_method_benchmark."
        "build_method_controllers",
        return_value={
            "m7_controller": object(),
            "m7_progress_monitor": object(),
            "m7_recovery_planner": object(),
        },
    ), patch(
        "scripts.run_controlled_three_method_benchmark."
        "build_experiment_context",
        return_value={},
    ) as context_builder, patch(
        "scripts.run_controlled_three_method_benchmark."
        "run_real_method",
        return_value=metrics,
    ) as runner:
        result = run_trial("M7", INSTANCE, ARGS, {"actor": None, "critic": None})

    runner.assert_called_once()
    assert context_builder.call_count == 1
    assert result["success"] is True
    assert result["makespan"] == 5
    assert result["path_length"] == 8
    assert result["episode_wall_time"] >= 0


def test_two_way_dispatch_and_metrics():
    hybrid = {
        "all_goals_reached": True,
        "collision": False,
        "deadlock": False,
        "planning_failed": False,
        "steps": 6,
        "path_length": 9,
        "planning_time": 0.15,
        "lns_steps": 4,
        "m7_steps": 2,
        "switch_events": [object(), object()],
        "return_replans": 1,
        "planning_failures": 0,
    }

    with patch(
        "scripts.run_controlled_three_method_benchmark."
        "build_method_controllers",
        return_value={
            "m7_controller": object(),
            "m7_progress_monitor": object(),
            "m7_recovery_planner": object(),
        },
    ), patch(
        "scripts.run_controlled_three_method_benchmark."
        "build_experiment_context",
        return_value={},
    ), patch(
        "scripts.run_controlled_three_method_benchmark."
        "run_two_way_episode",
        return_value=hybrid,
    ) as runner:
        result = run_trial(
            "TWO-WAY", INSTANCE, ARGS,
            {"actor": None, "critic": None},
        )

    runner.assert_called_once()
    assert result["success"] is True
    assert result["lns_steps"] == 4
    assert result["m7_steps"] == 2
    assert result["switch_count"] == 2
    assert result["return_replans"] == 1
    assert result["lns_planning_time"] == 0.15
