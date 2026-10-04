import time

from src.evaluation.metrics import create_episode_metrics
from src.evaluation.path_length import compute_step_path_length
from src.mapf.mdd_sat import (
    position_paths_to_actions,
    solve_minimum_makespan,
)


def run_mdd_sat_episode(
    simulator,
    max_makespan=100,
):
    starts = dict(simulator.agent_positions)
    goals = dict(simulator.agent_goals)

    planning_start = time.perf_counter()

    solution = solve_minimum_makespan(
        grid=simulator.grid,
        starts=starts,
        goals=goals,
        max_makespan=max_makespan,
    )

    planning_time = time.perf_counter() - planning_start

    if solution is None:
        return {
            "steps": 0,
            "collision": False,
            "all_goals_reached": False,
            "planning_failed": True,
        }

    action_paths = position_paths_to_actions(
        solution["paths"]
    )

    total_path_length = 0
    collision = False
    steps = 0

    for timestep in range(solution["makespan"]):
        actions = {
            agent_id: path[timestep]
            for agent_id, path in action_paths.items()
        }

        positions_before = dict(
            simulator.agent_positions
        )

        result = simulator.step_joint(actions)

        positions_after = dict(
            simulator.agent_positions
        )

        total_path_length += compute_step_path_length(
            positions_before,
            positions_after,
        )

        steps = timestep + 1
        collision = bool(
            result.get("collision", False)
        )

        if collision:
            break

    all_goals_reached = simulator.all_goals_reached()

    return {
        "steps": steps,
        "collision": collision,
        "all_goals_reached": all_goals_reached,
        "planning_failed": False,
        "metrics": create_episode_metrics(
            method="MDD-SAT",
            success=all_goals_reached,
            collision=collision,
            deadlock=False,
            makespan=steps,
            path_length=total_path_length,
            planning_time=planning_time,
            shield_interventions=0,
        ),
    }
