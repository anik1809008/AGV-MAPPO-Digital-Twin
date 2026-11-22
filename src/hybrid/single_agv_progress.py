"""Progress verification for experimental single-AGV recovery."""

from src.environment.actions import Action, ACTION_DELTAS


def verify_recovery_progress(
    simulator, plan, index, before, executed, collision,
):
    """Accept progress only when the expected recovery move occurred."""
    if collision or not plan.get("single_agv_completion"):
        return False

    agent = plan["active_agent"]
    route = plan["route_positions"]

    if not 0 <= index < len(route) - 1:
        return False

    after = simulator.agent_positions
    goals = simulator.agent_goals

    if set(before) != set(after) or set(before) != set(goals):
        return False

    if before[agent] != route[index]:
        return False

    if after[agent] != route[index + 1]:
        return False

    action = plan["actions"][agent][index]

    if executed.get(agent) != action or action == int(Action.WAIT):
        return False

    dx, dy = ACTION_DELTAS[Action(action)]
    if after[agent] != (
        before[agent][0] + dx,
        before[agent][1] + dy,
    ):
        return False

    # Finished agents must stay at their goals.
    for other in before:
        if other == agent:
            continue
        if before[other] != goals[other]:
            return False
        if after[other] != before[other]:
            return False
        if executed.get(other) != int(Action.WAIT):
            return False

    return True


def acknowledge_recovery_progress(
    simulator, plan, index, before, executed, collision,
):
    """Reset the counter only after a verified route advancement."""
    verified = verify_recovery_progress(
        simulator, plan, index, before, executed, collision,
    )

    if verified:
        simulator.non_progress_steps = 0

    return verified
