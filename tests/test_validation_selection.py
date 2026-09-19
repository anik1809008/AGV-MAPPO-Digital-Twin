from src.evaluation.validation_selection import (
    group_validation_candidates,
    load_validation_rows,
    parse_csv_bool,
    select_best_validation_candidate,
    summarize_validation_group,
)
def test_summarize_validation_group():
    rows = [
        {
            "collision": False,
            "deadlock": False,
            "success": True,
            "makespan": 10,
            "path_length": 20,
            "planning_time": 0.1,
        },
        {
            "collision": True,
            "deadlock": False,
            "success": False,
            "makespan": 20,
            "path_length": 30,
            "planning_time": 0.2,
        },
    ]

    summary = summarize_validation_group(
        rows
    )

    assert summary["collision_rate"] == 0.5
    assert summary["deadlock_rate"] == 0.0
    assert summary["success_rate"] == 0.5
    assert summary["mean_makespan"] == 15.0
    assert summary["mean_path_length"] == 25.0


def test_selection_prioritizes_collision_safety():
    candidates = {
        1: [
            {
                "collision": True,
                "deadlock": False,
                "success": True,
                "makespan": 5,
                "path_length": 10,
                "planning_time": 0.01,
            }
        ],
        5: [
            {
                "collision": False,
                "deadlock": False,
                "success": False,
                "makespan": 20,
                "path_length": 40,
                "planning_time": 0.1,
            }
        ],
    }

    result = select_best_validation_candidate(
        candidates
    )

    assert result["candidate"] == 5


def test_selection_uses_success_after_safety():
    candidates = {
        1: [
            {
                "collision": False,
                "deadlock": False,
                "success": False,
                "makespan": 10,
                "path_length": 20,
                "planning_time": 0.01,
            }
        ],
        5: [
            {
                "collision": False,
                "deadlock": False,
                "success": True,
                "makespan": 30,
                "path_length": 60,
                "planning_time": 0.1,
            }
        ],
    }

    result = select_best_validation_candidate(
        candidates
    )

    assert result["candidate"] == 5
def test_parse_csv_bool():
    assert parse_csv_bool("True") is True
    assert parse_csv_bool("False") is False


def test_load_and_group_validation_rows(tmp_path):
    path = tmp_path / "validation.csv"

    path.write_text(
        (
            "method,scenario_id,agent_count,seed,"
            "checkpoint_scenario_id,"
            "episodes_per_scenario,"
            "uncertainty_condition,"
            "latency_steps,"
            "immediate_probability,"
            "m5_threshold,"
            "success,collision,deadlock,"
            "makespan,path_length,"
            "planning_time,"
            "shield_interventions\n"
            "M6,16,8,0,,5,perfect,0,1.0,,"
            "True,False,False,10,20,0.1,0\n"
            "M6,17,8,0,,5,perfect,0,1.0,,"
            "False,False,False,20,30,0.2,0\n"
        ),
        encoding="utf-8",
    )

    rows = load_validation_rows(
        str(path)
    )

    assert rows[0]["success"] is True
    assert rows[0]["collision"] is False
    assert rows[0]["episodes_per_scenario"] == 5

    grouped = group_validation_candidates(
        rows=rows,
        method="M6",
        agent_count=8,
        seed=0,
    )

    assert 5 in grouped
    assert len(grouped[5]) == 2
