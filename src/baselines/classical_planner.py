from collections import deque

from src.environment.actions import Action


ACTION_ORDER = [
    Action.NORTH,
    Action.SOUTH,
    Action.EAST,
    Action.WEST,
]


def shortest_path(grid, start, goal):
    if start == goal:
        return []

    height = len(grid)
    width = len(grid[0])

    queue = deque([start])
    parent = {start: None}
    parent_action = {}

    while queue:
        x, y = queue.popleft()

        for action in ACTION_ORDER:
            if action == Action.NORTH:
                next_pos = (x, y - 1)
            elif action == Action.SOUTH:
                next_pos = (x, y + 1)
            elif action == Action.EAST:
                next_pos = (x + 1, y)
            else:
                next_pos = (x - 1, y)

            nx, ny = next_pos

            if not (
                0 <= nx < width
                and 0 <= ny < height
            ):
                continue

            if grid[ny][nx] != 0:
                continue

            if next_pos in parent:
                continue

            parent[next_pos] = (x, y)
            parent_action[next_pos] = action

            if next_pos == goal:
                path = []
                current = goal

                while current != start:
                    path.append(
                        parent_action[current]
                    )
                    current = parent[current]

                path.reverse()
                return path

            queue.append(next_pos)

    return None
