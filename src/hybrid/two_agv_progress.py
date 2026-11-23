"""Verify actual progress during sequential two-AGV recovery."""

from src.environment.actions import Action


def acknowledge_two_agv_progress(
    simulator, plan, index, before, executed, collision,
):
    if collision or not plan.get("two_agv_completion"):
        return False

    expected = plan.get("recovery_expected_positions")
    if expected is None or not 0 <= index < len(expected) - 1:
        return False

    actions = plan["actions"]
    after = simulator.agent_positions
    agents = set(before)

    if (
        agents != set(after)
        or agents != set(actions)
        or agents != set(executed)
        or agents != set(expected[index])
        or agents != set(expected[index + 1])
    ):
        return False

    planned = {agent: actions[agent][index] for agent in agents}
    moving = [
        agent for agent, action in planned.items()
        if action != int(Action.WAIT)
    ]

    if len(moving) != 1:
        return False

    active = moving[0]

    if (
        before != expected[index]
        or after != expected[index + 1]
        or executed != planned
        or after[active] == before[active]
    ):
        return False

    for agent in agents - {active}:
        if (
            planned[agent] != int(Action.WAIT)
            or after[agent] != before[agent]
        ):
            return False

    simulator.non_progress_steps = 0
    return True
