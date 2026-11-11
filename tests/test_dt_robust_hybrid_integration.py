from types import SimpleNamespace
from unittest.mock import patch

import pytest

import scripts.run_four_method_evaluation as evaluation
from src.hybrid.execution_aware_supervisor import (
    ExecutionAwareTwoWaySupervisor,
)


@pytest.mark.parametrize(
    "method,expected_robust",
    [
        ("DT-ONLY-HYBRID", False),
        ("DT-ROBUST-HYBRID", True),
        ("ONE-WAY-ROBUST-HYBRID", True),
        ("EXECUTION-AWARE-HYBRID", False),
    ],
)
def test_hybrid_configuration(method, expected_robust):
    controller = SimpleNamespace(enable_robust_progress=False)

    context = {
        "m7_controller": controller,
    }

    specification = {
        "seed": "0",
        "map_path": "dummy.map",
        "scenario_id": "16",
        "agent_count": "8",
        "start_index": "0",
        "latency_steps": "2",
        "immediate_probability": "0.8",
    }

    result = {
        "all_goals_reached": True,
        "collision": False,
        "deadlock": False,
        "steps": 20,
        "path_length": 100,
        "planning_time": 1.0,
        "return_replans": 0,
        "recovery_attempts": 0,
        "recovery_pause_steps": 0,
        "switch_events": [],
        "planning_failures": 0,
    }

    with (
        patch.object(
            evaluation,
            "get_random_scenario_path",
            return_value="dummy.scen",
        ),
        patch.object(
            evaluation,
            "build_training_instance",
            return_value={
                "grid": object(),
                "starts": {},
                "goals": {},
            },
        ),
        patch.object(
            evaluation,
            "build_method_controllers",
            return_value={
                "m7_controller": controller,
                "m7_progress_monitor": None,
                "m7_recovery_planner": None,
            },
        ),
        patch.object(
            evaluation,
            "build_experiment_context",
            return_value=context,
        ),
        patch.object(
            evaluation,
            "run_two_way_episode",
            return_value=result,
        ) as runner,
    ):
        metrics = evaluation.run_episode(
            method,
            specification,
            models={"actor": object(), "critic": object()},
        )

    assert metrics["success"] is True
    assert controller.enable_robust_progress is expected_robust

    kwargs = runner.call_args.kwargs

    assert isinstance(
        kwargs["supervisor"],
        ExecutionAwareTwoWaySupervisor,
    )

    assert kwargs["enable_execution_handover"] is True

    dt_mode = method != "EXECUTION-AWARE-HYBRID"

    assert kwargs["use_dt_observations"] is dt_mode
    assert kwargs["use_dt_inferred_centers"] is dt_mode
    assert kwargs["allow_return_to_lns"] is (method != "ONE-WAY-ROBUST-HYBRID")


def test_original_hybrid_runner_defaults_unchanged():
    from inspect import signature
    from src.hybrid.two_way_episode_runner import (
        run_two_way_episode,
    )

    parameters = signature(run_two_way_episode).parameters

    assert parameters["use_dt_observations"].default is False
    assert parameters["use_dt_inferred_centers"].default is False
