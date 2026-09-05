from src.baselines.multi_agent_classical import (
    plan_multi_agent_paths,
    get_joint_actions,
    resolve_joint_conflicts,
)
from src.environment.actions import ACTION_DELTAS, Action


def run_m1_episode(
    simulator,
    max_steps=200,
):
    starts = dict(simulator.agent_positions)
    goals = dict(simulator.agent_goals)

    paths = plan_multi_agent_paths(
        grid=simulator.grid,
        starts=starts,
        goals=goals,
    )

    if paths is None:
        return {
            "steps": 0,
            "collision": False,
            "all_goals_reached": False,
            "planning_failed": True,
        }

    collision = False
    all_goals_reached = False
    steps = 0

    for timestep in range(max_steps):
        proposed_actions = get_joint_actions(
            paths=paths,
            timestep=timestep,
        )

        current_positions = dict(
            simulator.agent_positions
        )

        proposed_next_positions = {}

        for agent_id, action in proposed_actions.items():
            dx, dy = ACTION_DELTAS[
                Action(action)
            ]

            candidate = (
                current_positions[agent_id][0] + dx,
                current_positions[agent_id][1] + dy,
            )

            if simulator.is_free(candidate):
                proposed_next_positions[agent_id] = candidate
            else:
                proposed_next_positions[agent_id] = (
                    current_positions[agent_id]
                )

        resolved_actions = resolve_joint_conflicts(
            current_positions=current_positions,
            proposed_next_positions=proposed_next_positions,
            proposed_actions=proposed_actions,
        )

        result = simulator.step_joint(
            resolved_actions
        )

        steps = timestep + 1
        collision = bool(
            result.get("collision", False)
        )

        all_goals_reached = (
            simulator.all_goals_reached()
        )

        if collision or all_goals_reached:
            break

    return {
        "steps": steps,
        "collision": collision,
        "all_goals_reached": all_goals_reached,
        "planning_failed": False,
    }
