from src.evaluation.metrics import create_episode_metrics


def test_create_episode_metrics():
    metrics = create_episode_metrics(
        method="M4",
        success=True,
        collision=False,
        deadlock=False,
        makespan=25,
        path_length=120,
        planning_time=0.03,
        shield_interventions=4,
    )

    assert metrics == {
        "method": "M4",
        "success": True,
        "collision": False,
        "deadlock": False,
        "makespan": 25,
        "path_length": 120,
        "planning_time": 0.03,
        "shield_interventions": 4,
    }
