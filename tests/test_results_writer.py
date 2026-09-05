import csv

from src.evaluation.results_writer import append_result_csv


def test_append_result_csv(tmp_path):
    path = tmp_path / "results.csv"


    metrics = {
        "method": "M4",
        "scenario_id": "random-1-000",
        "agent_count": 8,
        "latency_steps": 2,
        "m5_threshold": "",
        "success": True,
        "collision": False,
        "deadlock": False,
        "makespan": 20,
        "path_length": 80,
        "planning_time": 0.02,
        "shield_interventions": 3,
    }




    append_result_csv(
        str(path),
        metrics,
    )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        rows = list(
            csv.DictReader(file)
        )

    assert len(rows) == 1
    assert rows[0]["method"] == "M4"
    assert rows[0]["success"] == "True"
    assert rows[0]["makespan"] == "20"
    assert rows[0]["shield_interventions"] == "3"
    assert rows[0]["scenario_id"] == "random-1-000"
    assert rows[0]["agent_count"] == "8"
    assert rows[0]["latency_steps"] == "2"
    assert rows[0]["m5_threshold"] == ""
