from unittest.mock import patch

from scripts.run_four_method_evaluation import METHODS, run_episode


def test_all_four_methods_registered():
    assert METHODS == (
        "LNS-SAT",
        "M7",
        "TWO-WAY",
        "RECOVERY-HYBRID",
        "EXECUTION-AWARE-HYBRID",
        "EARLY-SWITCH-ONLY",
    )


def test_recovery_method_enables_pause():
    specification = {
        "map_path": "benchmarks/movingai/warehouse-10-20-10-2-1.map",
        "scenario_id": "18",
        "agent_count": "8",
        "start_index": "0",
        "seed": "1",
        "latency_steps": "2",
        "immediate_probability": "0.8",
    }

    instance = {
        "grid": [[0, 0, 0]],
        "starts": {0: (0, 0)},
        "goals": {0: (2, 0)},
    }

    controllers = {
        "m7_controller": object(),
        "m7_progress_monitor": object(),
        "m7_recovery_planner": object(),
    }

    result = {
        "all_goals_reached": True,
        "collision": False,
        "deadlock": False,
        "steps": 5,
        "path_length": 2,
        "planning_time": 0.1,
        "return_replans": 1,
        "recovery_attempts": 1,
        "recovery_pause_steps": 2,
        "switch_events": [],
        "planning_failures": 0,
    }

    with patch(
        "scripts.run_four_method_evaluation.build_training_instance",
        return_value=instance,
    ), patch(
        "scripts.run_four_method_evaluation.build_method_controllers",
        return_value=controllers,
    ), patch(
        "scripts.run_four_method_evaluation.build_experiment_context",
        return_value={},
    ), patch(
        "scripts.run_four_method_evaluation.run_two_way_episode",
        return_value=result,
    ) as runner:
        metrics = run_episode(
            "RECOVERY-HYBRID",
            specification,
            {"actor": None, "critic": None},
        )

    assert runner.call_args.kwargs["enable_recovery_pause"] is True
    assert metrics["recovery_attempts"] == 1
    assert metrics["recovery_pause_steps"] == 2


def test_episode_uses_specified_seed():
    """Episode execution must use its own evaluation seed."""
    specification = {
        "map_path": "benchmarks/movingai/warehouse-10-20-10-2-1.map",
        "scenario_id": "18",
        "agent_count": "8",
        "start_index": "0",
        "seed": "4",
        "latency_steps": "2",
        "immediate_probability": "0.8",
    }

    instance = {
        "grid": [[0, 0, 0]],
        "starts": {0: (0, 0)},
        "goals": {0: (2, 0)},
    }

    metrics = {
        "success": True,
        "collision": False,
        "deadlock": False,
        "makespan": 2,
        "path_length": 2,
        "planning_time": 0.1,
    }

    with patch(
        "scripts.run_four_method_evaluation.build_training_instance",
        return_value=instance,
    ), patch(
        "scripts.run_four_method_evaluation.run_lns_sat_uncertain_episode",
        return_value=metrics,
    ) as runner:
        result = run_episode("LNS-SAT", specification, None)

    assert runner.call_args.kwargs["seed"] == 4
    assert runner.call_args.kwargs["immediate_probability"] == 0.8
    assert result["success"] is True


def test_seed_specific_checkpoint_is_loaded_for_episode():
    """Checkpoint selection follows each episode's seed."""
    import sys
    from tempfile import TemporaryDirectory
    from pathlib import Path
    import csv

    from scripts.run_four_method_evaluation import FIELDS, main

    with TemporaryDirectory() as folder:
        matrix = Path(folder) / "matrix.csv"
        output = Path(folder) / "output.csv"

        with matrix.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "episode_id", "map_path", "scenario_id",
                    "agent_count", "start_index", "seed",
                    "latency_steps", "immediate_probability",
                ],
            )
            writer.writeheader()
            writer.writerow({
                "episode_id": 1,
                "map_path": "test.map",
                "scenario_id": 18,
                "agent_count": 8,
                "start_index": 0,
                "seed": 4,
                "latency_steps": 2,
                "immediate_probability": 0.8,
            })

        argv = [
            "evaluation",
            "--matrix", str(matrix),
            "--output", str(output),
            "--method", "M7",
            "--limit", "1",
        ]

        metrics = {
            "success": True,
            "collision": False,
            "deadlock": False,
            "makespan": 2,
            "path_length": 2,
            "planning_time": 0.1,
        }

        with patch.object(sys, "argv", argv), patch(
            "scripts.run_four_method_evaluation.load_validation_models",
            return_value={"actor": None, "critic": None},
        ) as loader, patch(
            "scripts.run_four_method_evaluation.run_episode",
            return_value=metrics,
        ) as runner:
            main()

            loader.assert_called_once_with(
                method="M7",
                agent_count=8,
                seed=4,
                episodes_per_scenario=20,
            )

            argv.append("--resume")
            with patch.object(sys, "argv", argv):
                main()

            assert runner.call_count == 1

        with output.open(newline="") as handle:
            assert len(list(csv.DictReader(handle))) == 1


def test_hybrid_methods_use_matched_conditions():
    specification = {
        "map_path": "benchmarks/movingai/warehouse-10-20-10-2-1.map",
        "scenario_id": "18",
        "agent_count": "8",
        "start_index": "0",
        "seed": "1",
        "latency_steps": "2",
        "immediate_probability": "0.8",
    }
    instance = {
        "grid": [[0, 0, 0]],
        "starts": {0: (0, 0)},
        "goals": {0: (2, 0)},
    }
    controllers = {
        "m7_controller": object(),
        "m7_progress_monitor": object(),
        "m7_recovery_planner": object(),
    }
    outcome = {
        "all_goals_reached": True,
        "collision": False,
        "deadlock": False,
        "steps": 5,
        "path_length": 2,
        "planning_time": 0.1,
        "return_replans": 0,
        "recovery_attempts": 0,
        "recovery_pause_steps": 0,
        "switch_events": [],
        "planning_failures": 0,
    }

    with patch(
        "scripts.run_four_method_evaluation.build_training_instance",
        return_value=instance,
    ), patch(
        "scripts.run_four_method_evaluation.build_method_controllers",
        return_value=controllers,
    ), patch(
        "scripts.run_four_method_evaluation.build_experiment_context",
        return_value={},
    ) as context_builder, patch(
        "scripts.run_four_method_evaluation.run_two_way_episode",
        return_value=outcome,
    ) as runner:
        for method, enabled in [
            ("TWO-WAY", False),
            ("RECOVERY-HYBRID", True),
        ]:
            run_episode(
                method,
                specification,
                {"actor": None, "critic": None},
            )

            assert runner.call_args.kwargs[
                "enable_recovery_pause"
            ] is enabled

            kwargs = context_builder.call_args.kwargs
            assert kwargs["latency_steps"] == 2
            assert kwargs["max_steps"] == 400
            assert kwargs["starts"] == instance["starts"]
            assert kwargs["goals"] == instance["goals"]
            assert (
                kwargs["execution_delay_model"].immediate_probability
                == 0.8
            )
