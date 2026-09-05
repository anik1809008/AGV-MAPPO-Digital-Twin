from src.evaluation.experiment_runner import run_experiment_sweep


def test_run_experiment_sweep(tmp_path):
    path = tmp_path / "sweep.csv"

    def fake_run_method(
        method,
        latency_steps,
    ):
        return {
            "method": method,
            "success": True,
            "collision": False,
            "deadlock": False,
            "makespan": 10,
            "path_length": 20,
            "planning_time": 0.01,
            "shield_interventions": 0,
        }

    results = run_experiment_sweep(
        methods=["M2", "M3"],
        latency_levels=[0, 1],
        run_method_fn=fake_run_method,
        results_path=str(path),
        scenario_id="test-001",
        agent_count=8,
    )

    assert len(results) == 4

    assert results[0]["method"] == "M2"
    assert results[0]["latency_steps"] == 0

    assert results[-1]["method"] == "M3"
    assert results[-1]["latency_steps"] == 1

    assert path.exists()
