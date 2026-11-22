"""DT-only route validation for experimental AGV recovery."""

from src.environment.actions import Action, ACTION_DELTAS


def build_expected_positions(starts, actions, horizon):
    if set(starts) != set(actions):
        raise ValueError("Incomplete joint-action plan")

    if any(len(seq) != horizon for seq in actions.values()):
        raise ValueError("Inconsistent action lengths")

    current = {
        agent: tuple(position)
        for agent, position in starts.items()
    }

    expected = [dict(current)]

    for index in range(horizon):
        following = {}

        for agent in current:
            action = Action(actions[agent][index])
            dx, dy = ACTION_DELTAS[action]
            x, y = current[agent]

            following[agent] = (x + dx, y + dy)

        expected.append(following)
        current = following

    return expected


def recovery_route_is_valid(plan, dt, index, pending_commands):
    expected = plan.get("recovery_expected_positions")

    if expected is None:
        return False

    if pending_commands or not 0 <= index < len(expected):
        return False

    actual = dt.get("possible_current_positions", {})
    ages = dt.get("aoi_values", {})
    target = expected[index]

    if set(actual) != set(target) or set(ages) != set(target):
        return False

    for agent, position in target.items():
        if ages[agent] != 0:
            return False

        candidates = actual[agent]

        if len(candidates) != 1:
            return False

        if tuple(next(iter(candidates))) != position:
            return False

    return True
