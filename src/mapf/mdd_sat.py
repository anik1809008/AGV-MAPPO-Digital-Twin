from collections import deque


MOVE_DELTAS = [
    (0, 0),
    (0, -1),
    (0, 1),
    (1, 0),
    (-1, 0),
]


def shortest_distances(grid, goal):
    height = len(grid)
    width = len(grid[0])

    distances = {goal: 0}
    queue = deque([goal])

    while queue:
        x, y = queue.popleft()

        for dx, dy in MOVE_DELTAS[1:]:
            nx = x + dx
            ny = y + dy

            if not (
                0 <= nx < width
                and 0 <= ny < height
            ):
                continue

            if grid[ny][nx] != 0:
                continue

            next_pos = (nx, ny)

            if next_pos in distances:
                continue

            distances[next_pos] = distances[(x, y)] + 1
            queue.append(next_pos)

    return distances


def build_mdd(grid, start, goal, horizon):
    distances_from_start = shortest_distances(
        grid=grid,
        goal=start,
    )

    distances_to_goal = shortest_distances(
        grid=grid,
        goal=goal,
    )

    if (
        start not in distances_to_goal
        or goal not in distances_from_start
    ):
        return None

    layers = []

    for timestep in range(horizon + 1):
        remaining = horizon - timestep
        layer = set()

        for position, distance_to_goal in distances_to_goal.items():
            distance_from_start = distances_from_start.get(position)

            if distance_from_start is None:
                continue

            if distance_from_start > timestep:
                continue

            if distance_to_goal > remaining:
                continue

            layer.add(position)

        layers.append(layer)

    if start not in layers[0]:
        return None

    if goal not in layers[-1]:
        return None

    return layers
def create_node_variables(mdds):
    variables = {}
    next_variable = 1

    for agent_id in sorted(mdds):
        mdd = mdds[agent_id]

        for timestep, layer in enumerate(mdd):
            for position in sorted(layer):
                key = (
                    agent_id,
                    timestep,
                    position,
                )

                variables[key] = next_variable
                next_variable += 1

    return variables
def add_exactly_one_position_constraints(
    cnf,
    mdds,
    variables,
):
    for agent_id in sorted(mdds):
        mdd = mdds[agent_id]

        for timestep, layer in enumerate(mdd):
            layer_variables = [
                variables[
                    (
                        agent_id,
                        timestep,
                        position,
                    )
                ]
                for position in sorted(layer)
            ]

            cnf.append(layer_variables)

            for i in range(len(layer_variables)):
                for j in range(i + 1, len(layer_variables)):
                    cnf.append(
                        [
                            -layer_variables[i],
                            -layer_variables[j],
                        ]
                    )
def add_movement_constraints(
    cnf,
    mdds,
    variables,
):
    for agent_id in sorted(mdds):
        mdd = mdds[agent_id]

        for timestep in range(len(mdd) - 1):
            current_layer = mdd[timestep]
            next_layer = mdd[timestep + 1]

            for position in sorted(current_layer):
                x, y = position

                allowed_next = []

                for dx, dy in MOVE_DELTAS:
                    candidate = (
                        x + dx,
                        y + dy,
                    )

                    if candidate in next_layer:
                        allowed_next.append(
                            variables[
                                (
                                    agent_id,
                                    timestep + 1,
                                    candidate,
                                )
                            ]
                        )

                current_variable = variables[
                    (
                        agent_id,
                        timestep,
                        position,
                    )
                ]

                cnf.append(
                    [-current_variable] + allowed_next
                )
def add_vertex_conflict_constraints(
    cnf,
    mdds,
    variables,
):
    agent_ids = sorted(mdds)

    horizon = len(mdds[agent_ids[0]])

    for timestep in range(horizon):
        position_users = {}

        for agent_id in agent_ids:
            for position in mdds[agent_id][timestep]:
                position_users.setdefault(
                    position,
                    [],
                ).append(
                    variables[
                        (
                            agent_id,
                            timestep,
                            position,
                        )
                    ]
                )

        for variable_list in position_users.values():
            for i in range(len(variable_list)):
                for j in range(i + 1, len(variable_list)):
                    cnf.append(
                        [
                            -variable_list[i],
                            -variable_list[j],
                        ]
                    )
def add_edge_swap_constraints(
    cnf,
    mdds,
    variables,
):
    agent_ids = sorted(mdds)
    horizon = len(mdds[agent_ids[0]])

    for timestep in range(horizon - 1):
        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                a = agent_ids[i]
                b = agent_ids[j]

                for a_from in mdds[a][timestep]:
                    for a_to in mdds[a][timestep + 1]:
                        if a_from == a_to:
                            continue

                        for b_from in mdds[b][timestep]:
                            for b_to in mdds[b][timestep + 1]:
                                if b_from == b_to:
                                    continue

                                if (
                                    a_from == b_to
                                    and a_to == b_from
                                ):
                                    cnf.append(
                                        [
                                            -variables[
                                                (
                                                    a,
                                                    timestep,
                                                    a_from,
                                                )
                                            ],
                                            -variables[
                                                (
                                                    a,
                                                    timestep + 1,
                                                    a_to,
                                                )
                                            ],
                                            -variables[
                                                (
                                                    b,
                                                    timestep,
                                                    b_from,
                                                )
                                            ],
                                            -variables[
                                                (
                                                    b,
                                                    timestep + 1,
                                                    b_to,
                                                )
                                            ],
                                        ]
                                    )
def solve_fixed_makespan(
    grid,
    starts,
    goals,
    makespan,
):
    from pysat.formula import CNF
    from pysat.solvers import Minisat22

    mdds = {}

    for agent_id in sorted(starts):
        mdd = build_mdd(
            grid=grid,
            start=starts[agent_id],
            goal=goals[agent_id],
            horizon=makespan,
        )

        if mdd is None:
            return None

        mdds[agent_id] = mdd

    variables = create_node_variables(mdds)
    cnf = CNF()

    add_exactly_one_position_constraints(
        cnf=cnf,
        mdds=mdds,
        variables=variables,
    )

    add_movement_constraints(
        cnf=cnf,
        mdds=mdds,
        variables=variables,
    )

    add_vertex_conflict_constraints(
        cnf=cnf,
        mdds=mdds,
        variables=variables,
    )

    add_edge_swap_constraints(
        cnf=cnf,
        mdds=mdds,
        variables=variables,
    )

    with Minisat22(
        bootstrap_with=cnf.clauses
    ) as solver:
        if not solver.solve():
            return None

        model = set(
            literal
            for literal in solver.get_model()
            if literal > 0
        )

    paths = {}

    for agent_id in sorted(mdds):
        path = []

        for timestep, layer in enumerate(
            mdds[agent_id]
        ):
            selected_position = None

            for position in sorted(layer):
                variable = variables[
                    (
                        agent_id,
                        timestep,
                        position,
                    )
                ]

                if variable in model:
                    selected_position = position
                    break

            if selected_position is None:
                return None

            path.append(selected_position)

        paths[agent_id] = path

    return paths
def solve_minimum_makespan(
    grid,
    starts,
    goals,
    max_makespan=100,
):
    lower_bound = 0

    for agent_id in sorted(starts):
        distances = shortest_distances(
            grid=grid,
            goal=goals[agent_id],
        )

        if starts[agent_id] not in distances:
            return None

        lower_bound = max(
            lower_bound,
            distances[starts[agent_id]],
        )

    for makespan in range(
        lower_bound,
        max_makespan + 1,
    ):
        paths = solve_fixed_makespan(
            grid=grid,
            starts=starts,
            goals=goals,
            makespan=makespan,
        )

        if paths is not None:
            return {
                "makespan": makespan,
                "paths": paths,
            }

    return None
