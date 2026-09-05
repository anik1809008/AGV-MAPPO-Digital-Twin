from src.baselines.classical_planner import shortest_path
from src.environment.actions import Action


def plan_multi_agent_paths(
    grid,
    starts,
    goals,
):
    paths = {}

    for agent_id in starts:
        path = shortest_path(
            grid=grid,
            start=starts[agent_id],
            goal=goals[agent_id],
        )

        if path is None:
            return None

        paths[agent_id] = path

    return paths


def get_joint_actions(
    paths,
    timestep,
):
    actions = {}

    for agent_id, path in paths.items():
        if timestep < len(path):
            actions[agent_id] = path[timestep]
        else:
            actions[agent_id] = Action.WAIT

    return actions
def has_joint_conflict(
    current_positions,
    next_positions,
):
    seen = set()

    # Vertex conflict
    for position in next_positions.values():
        if position in seen:
            return True
        seen.add(position)

    # Edge-swap conflict
    agent_ids = list(current_positions.keys())

    for i in range(len(agent_ids)):
        for j in range(i + 1, len(agent_ids)):
            a = agent_ids[i]
            b = agent_ids[j]

            if (
                next_positions[a] == current_positions[b]
                and next_positions[b] == current_positions[a]
            ):
                return True

    return False
