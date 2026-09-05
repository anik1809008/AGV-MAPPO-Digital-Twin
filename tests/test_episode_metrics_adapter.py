from src.evaluation.episode_metrics_adapter import (
    build_marl_episode_metrics,
)


def test_build_marl_episode_metrics():
    episode_result = {
        "all_goals_reached": True,
        "collision": False,
        "deadlock": False,
        "makespan": 20,
        "path_length": 80,
    }

    metrics = build_marl_episode_metrics(
        method="M4",
        episode_result=episode_result,
        planning_time=0.02,
        shield_interventions=3,
    )

    assert metrics == {
        "method": "M4",
        "success": True,
        "collision": False,
        "deadlock": False,
        "makespan": 20,
        "path_length": 80,
        "planning_time": 0.02,
        "shield_interventions": 3,
    }
