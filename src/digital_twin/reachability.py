from src.environment.actions import Action, ACTION_DELTAS


def is_free(grid, position):
    x, y = position

    if y < 0 or y >= len(grid):
        return False

    if x < 0 or x >= len(grid[0]):
        return False

    return grid[y][x] == 0


def apply_action(grid, position, action):
    dx, dy = ACTION_DELTAS[Action(action)]

    next_position = (
        position[0] + dx,
        position[1] + dy,
    )

    if is_free(grid, next_position):
        return next_position

    return position


def compute_reachable_occupancy(
    grid,
    last_trusted_position,
    command_history,
):
    reachable = {last_trusted_position}

    for command in command_history:
        next_reachable = set(reachable)

        for position in reachable:
            moved_position = apply_action(
                grid,
                position,
                command,
            )

            next_reachable.add(moved_position)

        reachable = next_reachable

    return reachable
