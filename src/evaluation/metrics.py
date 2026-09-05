import time


def create_episode_metrics(
    method,
    success,
    collision,
    deadlock,
    makespan,
    path_length,
    planning_time,
    shield_interventions=0,
):
    return {
        "method": method,
        "success": bool(success),
        "collision": bool(collision),
        "deadlock": bool(deadlock),
        "makespan": int(makespan),
        "path_length": int(path_length),
        "planning_time": float(planning_time),
        "shield_interventions": int(
            shield_interventions
        ),
    }


def measure_planning_time(function, *args, **kwargs):
    start = time.perf_counter()

    result = function(
        *args,
        **kwargs,
    )

    elapsed = time.perf_counter() - start

    return result, elapsed
