"""Experimental sequential recovery for exactly two unfinished AGVs."""

from itertools import permutations

from src.hybrid.single_agv_completion import plan_single_agv_completion
from src.environment.actions import Action


def plan_two_agv_completion(grid, positions, goals):
    """Return a complete joint-action plan, or None.

    The input positions must come from verified Digital Twin observations.
    Exactly one AGV moves at a time.
    """
    if set(positions) != set(goals):
        return None

    current_positions = {
        agent: tuple(position)
        for agent, position in positions.items()
    }
    goals = {
        agent: tuple(goal)
        for agent, goal in goals.items()
    }

    unfinished = [
        agent for agent in goals
        if current_positions[agent] != goals[agent]
    ]

    if len(unfinished) != 2:
        return None

    best_plan = None

    for order in permutations(sorted(unfinished)):
        current = dict(current_positions)
        stages = []
        valid = True

        for active in order:
            temporary_goals = dict(current)
            temporary_goals[active] = goals[active]

            result = plan_single_agv_completion(
                grid, current, temporary_goals
            )

            if result is None or result[0] != active:
                valid = False
                break

            actions = result[1]
            stages.append((active, actions))
            current[active] = goals[active]

        if not valid:
            continue

        total_steps = sum(len(actions) for _, actions in stages)

        if best_plan is None or total_steps < best_plan["horizon"]:
            joint_actions = {
                agent: [int(Action.WAIT)] * total_steps
                for agent in current_positions
            }

            offset = 0
            for active, actions in stages:
                joint_actions[active][
                    offset:offset + len(actions)
                ] = actions
                offset += len(actions)

            best_plan = {
                "actions": joint_actions,
                "horizon": total_steps,
                "offset": 0,
                "two_agv_completion": True,
                "order": order,
                "stages": [
                    (agent, len(actions))
                    for agent, actions in stages
                ],
            }

    return best_plan
