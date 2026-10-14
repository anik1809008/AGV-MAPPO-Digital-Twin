from src.mapf.mdd_sat import (
    shortest_distances,
    solve_fixed_makespan,
)
def independent_shortest_paths(
    grid,
    starts,
    goals,
):
    paths = {}

    for agent_id in sorted(starts):
        distances = shortest_distances(
            grid=grid,
            goal=goals[agent_id],
        )

        start = starts[agent_id]

        if start not in distances:
            return None

        current = start
        path = [current]

        while current != goals[agent_id]:
            x, y = current
            current_distance = distances[current]

            candidates = [
                (x, y - 1),
                (x, y + 1),
                (x + 1, y),
                (x - 1, y),
            ]

            next_position = None

            for candidate in candidates:
                if (
                    candidate in distances
                    and distances[candidate]
                    == current_distance - 1
                ):
                    next_position = candidate
                    break

            if next_position is None:
                return None

            path.append(next_position)
            current = next_position

        paths[agent_id] = path

    return paths
def pad_paths_to_makespan(paths):
    makespan = max(
        len(path) - 1
        for path in paths.values()
    )

    padded = {}

    for agent_id, path in paths.items():
        padded_path = list(path)

        while len(padded_path) < makespan + 1:
            padded_path.append(padded_path[-1])

        padded[agent_id] = padded_path

    return padded
def find_earliest_conflict(paths):
    if not paths:
        return None

    agent_ids = sorted(paths)
    horizon = len(paths[agent_ids[0]])

    for timestep in range(horizon):
        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                a = agent_ids[i]
                b = agent_ids[j]

                if paths[a][timestep] == paths[b][timestep]:
                    return {
                        "type": "vertex",
                        "timestep": timestep,
                        "agents": (a, b),
                        "position": paths[a][timestep],
                    }

                if timestep == 0:
                    continue

                if (
                    paths[a][timestep - 1]
                    == paths[b][timestep]
                    and paths[b][timestep - 1]
                    == paths[a][timestep]
                ):
                    return {
                        "type": "swap",
                        "timestep": timestep - 1,
                        "agents": (a, b),
                        "positions": (
                            paths[a][timestep - 1],
                            paths[a][timestep],
                        ),
                    }

    return None
def build_repair_window(
    conflict,
    horizon,
    radius=2,
):
    start_t = max(
        0,
        conflict["timestep"] - radius,
    )

    end_t = min(
        horizon - 1,
        conflict["timestep"] + radius + 1,
    )

    return {
        "start_t": start_t,
        "end_t": end_t,
        "agents": conflict["agents"],
    }
def build_local_repair_problem(
    paths,
    window,
):
    starts = {}
    goals = {}

    start_t = window["start_t"]
    end_t = window["end_t"]

    for agent_id in window["agents"]:
        starts[agent_id] = paths[agent_id][start_t]
        goals[agent_id] = paths[agent_id][end_t]

    return {
        "starts": starts,
        "goals": goals,
        "horizon": end_t - start_t,
    }
def solve_local_repair(
    grid,
    paths,
    window,
    conflict=None,
    spatial_radius=3,
):
    problem = build_local_repair_problem(
        paths=paths,
        window=window,
    )

    local_grid = grid

    if conflict is not None:
        zone = build_spatial_zone(
            conflict=conflict,
            grid=grid,
            radius=spatial_radius,
        )

        required_positions = set(
            problem["starts"].values()
        )
        required_positions.update(
            problem["goals"].values()
        )

        local_grid = build_zone_restricted_grid(
            grid=grid,
            zone=zone,
            required_positions=required_positions,
        )

    final_goals = {
        agent_id: paths[agent_id][-1]
        for agent_id in window["agents"]
    }

    repaired = solve_fixed_makespan(
        grid=local_grid,
        starts=problem["starts"],
        goals=problem["goals"],
        makespan=problem["horizon"],
        absorbing_goals=final_goals,
    )

    return repaired
    problem = build_local_repair_problem(
        paths=paths,
        window=window,
    )
    final_goals = {
        agent_id: paths[agent_id][-1]
        for agent_id in window["agents"]
    }

    repaired = solve_fixed_makespan(
        grid=grid,
        starts=problem["starts"],
        goals=problem["goals"],
        makespan=problem["horizon"],
        absorbing_goals=final_goals,
    )
    return repaired
def expand_repair_window(
    window,
    horizon,
    amount=1,
):
    return {
        "start_t": max(
            0,
            window["start_t"] - amount,
        ),
        "end_t": min(
            horizon - 1,
            window["end_t"] + amount,
        ),
        "agents": window["agents"],
    }
def solve_local_repair_with_expansion(
    grid,
    paths,
    window,
    conflict=None,
    spatial_radius=3,
    max_spatial_radius=20,
    max_expansions=20,
):
    horizon = len(next(iter(paths.values())))

    for current_spatial_radius in range(
        spatial_radius,
        max_spatial_radius + 1,
    ):
        current_window = dict(window)

        for _ in range(max_expansions + 1):
            repaired = solve_local_repair(
                grid=grid,
                paths=paths,
                window=current_window,
                conflict=conflict,
                spatial_radius=current_spatial_radius,
            )

            if repaired is not None:
                return {
                    "window": current_window,
                    "paths": repaired,
                    "spatial_radius": current_spatial_radius,
                }

            current_window = expand_repair_window(
                window=current_window,
                horizon=horizon,
                amount=1,
            )

    return None
    horizon = len(next(iter(paths.values())))
    current_window = dict(window)

    for _ in range(max_expansions + 1):
        repaired = solve_local_repair(
            grid=grid,
            paths=paths,
            window=current_window,
            conflict=conflict,
            spatial_radius=spatial_radius,
        )

        if repaired is not None:
            return {
                "window": current_window,
                "paths": repaired,
            }

        current_window = expand_repair_window(
            window=current_window,
            horizon=horizon,
            amount=1,
        )

    return None
    horizon = len(next(iter(paths.values())))
    current_window = dict(window)

    for _ in range(max_expansions + 1):
        repaired = solve_local_repair(
            grid=grid,
            paths=paths,
            window=current_window,
        )

        if repaired is not None:
            return {
                "window": current_window,
                "paths": repaired,
            }

        current_window = expand_repair_window(
            window=current_window,
            horizon=horizon,
            amount=1,
        )

    return None
def apply_local_repair(
    paths,
    repair_result,
):
    updated = {
        agent_id: list(path)
        for agent_id, path in paths.items()
    }

    window = repair_result["window"]
    repaired_paths = repair_result["paths"]

    start_t = window["start_t"]
    end_t = window["end_t"]

    for agent_id, repaired_path in repaired_paths.items():
        updated[agent_id][
            start_t:end_t + 1
        ] = repaired_path

    return updated
def repair_earliest_conflict(
    grid,
    paths,
    radius=1,
    max_expansions=20,
):
    conflict = find_earliest_conflict(paths)

    if conflict is None:
        return {
            "paths": paths,
            "conflict": None,
            "repaired": False,
        }

    horizon = len(next(iter(paths.values())))

    window = build_repair_window(
        conflict=conflict,
        horizon=horizon,
        radius=radius,
    )
    repair_result = solve_local_repair_with_expansion(
        grid=grid,
        paths=paths,
        window=window,
        conflict=conflict,
        max_expansions=max_expansions,
    )
    if repair_result is None:
        return None

    updated_paths = apply_local_repair(
        paths=paths,
        repair_result=repair_result,
    )

    return {
        "paths": updated_paths,
        "conflict": conflict,
        "repaired": True,
    }
def repair_until_conflict_free(
    grid,
    paths,
    radius=1,
    max_expansions=20,
    max_iterations=100,
):
    current_paths = {
        agent_id: list(path)
        for agent_id, path in paths.items()
    }

    for iteration in range(max_iterations):
        conflict = find_earliest_conflict(
            current_paths
        )

        if conflict is None:
            return {
                "paths": current_paths,
                "iterations": iteration,
            }

        result = repair_earliest_conflict(
            grid=grid,
            paths=current_paths,
            radius=radius,
            max_expansions=max_expansions,
        )

        if result is None:
            return None

        current_paths = result["paths"]

    return None
def solve_lns_sat(
    grid,
    starts,
    goals,
    radius=1,
    max_expansions=20,
    max_iterations=100,
    max_makespan_increase=10,
):
    initial_paths = independent_shortest_paths(
        grid=grid,
        starts=starts,
        goals=goals,
    )

    if initial_paths is None:
        return None

    base_paths = pad_paths_to_makespan(
        initial_paths
    )

    initial_makespan = (
        len(next(iter(base_paths.values()))) - 1
    )

    for increase in range(
        max_makespan_increase + 1
    ):
        makespan = initial_makespan + increase

        candidate_paths = extend_paths_to_makespan(
            paths=base_paths,
            makespan=makespan,
        )

        repaired = repair_until_conflict_free(
            grid=grid,
            paths=candidate_paths,
            radius=radius,
            max_expansions=max_expansions,
            max_iterations=max_iterations,
        )

        if repaired is not None:
            paths = repaired["paths"]

            return {
                "paths": paths,
                "makespan": makespan,
                "iterations": repaired["iterations"],
            }

    return None
def extend_paths_to_makespan(
    paths,
    makespan,
):
    extended = {}

    for agent_id, path in paths.items():
        extended_path = list(path)

        while len(extended_path) < makespan + 1:
            extended_path.append(
                extended_path[-1]
            )

        extended[agent_id] = extended_path

    return extended
def build_spatial_zone(
    conflict,
    grid,
    radius=3,
):
    height = len(grid)
    width = len(grid[0])

    if conflict["type"] == "vertex":
        centers = [
            conflict["position"],
        ]
    else:
        centers = list(
            conflict["positions"]
        )

    zone = set()

    for cx, cy in centers:
        for y in range(
            max(0, cy - radius),
            min(height, cy + radius + 1),
        ):
            for x in range(
                max(0, cx - radius),
                min(width, cx + radius + 1),
            ):
                if grid[y][x] != 0:
                    continue

                if (
                    abs(x - cx)
                    + abs(y - cy)
                    <= radius
                ):
                    zone.add((x, y))

    return zone
def build_zone_restricted_grid(
    grid,
    zone,
    required_positions=(),
):
    allowed = set(zone)
    allowed.update(required_positions)

    restricted = []

    for y, row in enumerate(grid):
        restricted_row = []

        for x, cell in enumerate(row):
            position = (x, y)

            if (
                cell == 0
                and position in allowed
            ):
                restricted_row.append(0)
            else:
                restricted_row.append(1)

        restricted.append(restricted_row)

    return restricted
