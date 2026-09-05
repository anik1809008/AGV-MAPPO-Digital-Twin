from src.evaluation.metrics import create_episode_metrics


def build_marl_episode_metrics(
    method,
    episode_result,
    planning_time,
    shield_interventions=0,
):
    success = episode_result[
        "all_goals_reached"
    ]

    return create_episode_metrics(
        method=method,
        success=success,
        collision=episode_result["collision"],
        deadlock=episode_result["deadlock"],
        makespan=episode_result["makespan"],
        path_length=episode_result["path_length"],
        planning_time=planning_time,
        shield_interventions=shield_interventions,
    )
