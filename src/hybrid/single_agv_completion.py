"""Experimental completion planning for exactly one unfinished AGV."""

from collections import deque

from src.environment.actions import Action, ACTION_DELTAS


MOVES = (
    Action.NORTH,
    Action.SOUTH,
    Action.EAST,
    Action.WEST,
)


def plan_single_agv_completion(grid, positions, goals):
    """Return (agent_id, actions) or None.

    All positions must come from verified DT observations.
    Other agents are treated as stationary obstacles.
    """
    if set(positions) != set(goals):
        return None

    positions = {
        agent: tuple(pos) for agent, pos in positions.items()
    }
    goals = {
        agent: tuple(goal) for agent, goal in goals.items()
    }

    unfinished = [
        agent for agent in goals
        if positions[agent] != goals[agent]
    ]

    if len(unfinished) != 1:
        return None

    agent = unfinished[0]
    start = positions[agent]
    goal = goals[agent]

    blocked = {
        positions[a]
        for a in positions
        if a != agent
    }

    if start in blocked or goal in blocked:
        return None

    height = len(grid)
    width = len(grid[0])

    queue = deque([start])
    parent = {start: None}

    while queue:
        current = queue.popleft()

        if current == goal:
            break

        for action in MOVES:
            dx, dy = ACTION_DELTAS[action]
            nxt = (
                current[0] + dx,
                current[1] + dy,
            )
            x, y = nxt

            if not (0 <= x < width and 0 <= y < height):
                continue
            if grid[y][x] != 0:
                continue
            if nxt in blocked or nxt in parent:
                continue

            parent[nxt] = (current, action)
            queue.append(nxt)

    if goal not in parent:
        return None

    actions = []
    current = goal

    while current != start:
        previous, action = parent[current]
        actions.append(int(action))
        current = previous

    actions.reverse()

    return agent, actions
