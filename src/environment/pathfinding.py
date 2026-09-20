from collections import deque

from src.environment.actions import Action, ACTION_DELTAS


MOVEMENT_ACTIONS = (
    Action.NORTH,
    Action.SOUTH,
    Action.EAST,
    Action.WEST,
)


def shortest_path_distance(
    grid,
    start,
    goal,
):
    if start == goal:
        return 0

    height = len(grid)
    width = len(grid[0])

    queue = deque([
        (start, 0),
    ])
    visited = {start}

    while queue:
        position, distance = queue.popleft()

        for action in MOVEMENT_ACTIONS:
            dx, dy = ACTION_DELTAS[action]

            next_position = (
                position[0] + dx,
                position[1] + dy,
            )

            x, y = next_position

            if not (
                0 <= x < width
                and 0 <= y < height
            ):
                continue

            if grid[y][x] != 0:
                continue

            if next_position in visited:
                continue

            if next_position == goal:
                return distance + 1

            visited.add(next_position)

            queue.append(
                (
                    next_position,
                    distance + 1,
                )
            )

    return None
