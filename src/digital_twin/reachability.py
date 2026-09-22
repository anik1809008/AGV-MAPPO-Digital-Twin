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
    goal=None,
):
    reachable = {last_trusted_position}

    current_position = last_trusted_position

    for command_entry in command_history:
        if isinstance(command_entry, tuple):
            _, command = command_entry
        else:
            command = command_entry

        if goal is not None and current_position == goal:
            next_position = current_position
        else:
            next_position = apply_action(
                grid,
                current_position,
                command,
            )

        current_position = next_position
        reachable.add(current_position)

    return reachable

def compute_possible_transitions(
    grid,
    last_trusted_position,
    command_history,
    goal=None,
):
    transitions = {
        (
            last_trusted_position,
            last_trusted_position,
        )
    }

    current_position = last_trusted_position

    for command_entry in command_history:
        if isinstance(command_entry, tuple):
            _, command = command_entry
        else:
            command = command_entry

        if goal is not None and current_position == goal:
            next_position = current_position
        else:
            next_position = apply_action(
                grid,
                current_position,
                command,
            )

        transitions.add(
            (
                current_position,
                next_position,
            )
        )

        transitions.add(
            (
                next_position,
                next_position,
            )
        )

        current_position = next_position

    return transitions
