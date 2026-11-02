"""Open-loop LNS-SAT execution with actuator delays.

This baseline receives the initial state and plan, but does not replan
or receive additional telemetry during execution.
"""

import time

from src.environment.delayed_execution import DelayedCommandExecutor
from src.environment.execution_delay import ExecutionDelayModel
from src.environment.simulator import GroundTruthSimulator
from src.evaluation.path_length import compute_step_path_length
from src.mapf.lns_sat import solve_lns_sat
from src.mapf.mdd_sat import position_paths_to_actions


def run_lns_sat_uncertain_episode(
    grid,
    starts,
    goals,
    immediate_probability=1.0,
    seed=1,
    max_steps=400,
    max_iterations=100,
):
    simulator = GroundTruthSimulator(
        grid=grid,
        agent_positions=dict(starts),
        agent_goals=dict(goals),
    )
    executor = DelayedCommandExecutor(
        ExecutionDelayModel(
            immediate_probability=immediate_probability,
            seed=seed,
        )
    )

    start_time = time.perf_counter()
    solution = solve_lns_sat(
        grid=grid,
        starts=dict(starts),
        goals=dict(goals),
        max_iterations=max_iterations,
    )
    planning_time = time.perf_counter() - start_time

    result = {
        "success": False,
        "collision": False,
        "deadlock": False,
        "planning_failed": solution is None,
        "makespan": 0,
        "path_length": 0,
        "planning_time": planning_time,
    }

    if solution is None:
        return result

    action_paths = position_paths_to_actions(solution["paths"])

    for step in range(max_steps):
        if simulator.all_goals_reached():
            break

        commands = {
            agent_id: (
                actions[step]
                if step < len(actions)
                else 0
            )
            for agent_id, actions in action_paths.items()
        }

        executor.queue_commands(
            actions=commands,
            current_timestep=step,
        )
        ready = executor.get_ready_actions(current_timestep=step)

        executed = {
            agent_id: ready.get(agent_id, 0)
            for agent_id in starts
        }

        before = dict(simulator.agent_positions)
        outcome = simulator.step_joint(executed)

        result["path_length"] += compute_step_path_length(
            before,
            simulator.agent_positions,
        )
        result["makespan"] = step + 1
        result["collision"] = bool(outcome["collision"])
        result["deadlock"] = simulator.is_deadlocked()

        if result["collision"] or result["deadlock"]:
            break

    result["success"] = simulator.all_goals_reached()
    return result
