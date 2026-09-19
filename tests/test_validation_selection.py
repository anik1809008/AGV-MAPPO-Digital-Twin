from src.evaluation.validation_selection import (
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
