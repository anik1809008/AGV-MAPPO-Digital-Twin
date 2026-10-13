from src.mapf.lns_sat import independent_shortest_paths
def test_independent_shortest_paths():
    grid = [
        [0, 0, 0],
        [0, 0, 0],
    ]

    paths = independent_shortest_paths(
        grid=grid,
        starts={
            0: (0, 0),
            1: (2, 1),
        },
        goals={
            0: (2, 0),
            1: (0, 1),
        },
    )

    assert paths is not None
    assert paths[0][0] == (0, 0)
    assert paths[0][-1] == (2, 0)
    assert paths[1][0] == (2, 1)
    assert paths[1][-1] == (0, 1)
def test_pad_paths_to_common_makespan():
    from src.mapf.lns_sat import pad_paths_to_makespan

    paths = {
        0: [
            (0, 0),
            (1, 0),
            (2, 0),
        ],
        1: [
            (0, 1),
            (1, 1),
        ],
    }

    padded = pad_paths_to_makespan(paths)

    assert len(padded[0]) == 3
    assert len(padded[1]) == 3
    assert padded[1][-1] == (1, 1)
def test_find_earliest_vertex_conflict():
    from src.mapf.lns_sat import find_earliest_conflict

    paths = {
        0: [
            (0, 0),
            (1, 0),
            (2, 0),
        ],
        1: [
            (2, 0),
            (1, 0),
            (0, 0),
        ],
    }

    conflict = find_earliest_conflict(paths)

    assert conflict is not None
    assert conflict["type"] == "vertex"
    assert conflict["timestep"] == 1
    assert conflict["agents"] == (0, 1)
def test_find_earliest_swap_conflict():
    from src.mapf.lns_sat import find_earliest_conflict

    paths = {
        0: [
            (0, 0),
            (1, 0),
        ],
        1: [
            (1, 0),
            (0, 0),
        ],
    }

    conflict = find_earliest_conflict(paths)

    assert conflict is not None
    assert conflict["type"] == "swap"
    assert conflict["timestep"] == 0
    assert conflict["agents"] == (0, 1)
def test_build_repair_window():
    from src.mapf.lns_sat import build_repair_window

    conflict = {
        "type": "vertex",
        "timestep": 5,
        "agents": (0, 1),
        "position": (3, 3),
    }

    window = build_repair_window(
        conflict=conflict,
        horizon=12,
        radius=2,
    )

    assert window == {
        "start_t": 3,
        "end_t": 8,
        "agents": (0, 1),
    }
def test_build_local_repair_problem():
    from src.mapf.lns_sat import build_local_repair_problem

    paths = {
        0: [
            (0, 0),
            (1, 0),
            (2, 0),
            (3, 0),
        ],
        1: [
            (0, 1),
            (1, 1),
            (2, 1),
            (3, 1),
        ],
    }

    window = {
        "start_t": 1,
        "end_t": 3,
        "agents": (0, 1),
    }

    problem = build_local_repair_problem(
        paths=paths,
        window=window,
    )

    assert problem["starts"] == {
        0: (1, 0),
        1: (1, 1),
    }

    assert problem["goals"] == {
        0: (3, 0),
        1: (3, 1),
    }

    assert problem["horizon"] == 2
def test_solve_local_repair_detects_tight_unsat_window():
    from src.mapf.lns_sat import solve_local_repair

    grid = [
        [0, 0, 0],
        [0, 0, 0],
    ]

    paths = {
        0: [
            (0, 0),
            (1, 0),
            (2, 0),
        ],
        1: [
            (2, 0),
            (1, 0),
            (0, 0),
        ],
    }

    window = {
        "start_t": 0,
        "end_t": 2,
        "agents": (0, 1),
    }

    repaired = solve_local_repair(
        grid=grid,
        paths=paths,
        window=window,
    )

    assert repaired is None
def test_expand_repair_window():
    from src.mapf.lns_sat import expand_repair_window

    window = {
        "start_t": 3,
        "end_t": 6,
        "agents": (0, 1),
    }

    expanded = expand_repair_window(
        window=window,
        horizon=12,
        amount=2,
    )

    assert expanded == {
        "start_t": 1,
        "end_t": 8,
        "agents": (0, 1),
    }
def test_solve_local_repair_with_expansion():
    from src.mapf.lns_sat import (
        solve_local_repair_with_expansion,
    )

    grid = [
        [0, 0, 0],
        [0, 0, 0],
    ]

    paths = {
        0: [
            (0, 0),
            (1, 0),
            (2, 0),
            (2, 0),
            (2, 0),
        ],
        1: [
            (2, 0),
            (1, 0),
            (0, 0),
            (0, 0),
            (0, 0),
        ],
    }

    window = {
        "start_t": 1,
        "end_t": 2,
        "agents": (0, 1),
    }

    result = solve_local_repair_with_expansion(
        grid=grid,
        paths=paths,
        window=window,
        max_expansions=2,
    )

    assert result is not None
def test_apply_local_repair():
    from src.mapf.lns_sat import apply_local_repair

    paths = {
        0: [
            (0, 0),
            (1, 0),
            (2, 0),
            (3, 0),
        ],
    }

    repair_result = {
        "window": {
            "start_t": 1,
            "end_t": 2,
            "agents": (0,),
        },
        "paths": {
            0: [
                (1, 0),
                (1, 1),
            ],
        },
    }

    updated = apply_local_repair(
        paths=paths,
        repair_result=repair_result,
    )

    assert updated[0] == [
        (0, 0),
        (1, 0),
        (1, 1),
        (3, 0),
    ]
def test_repair_earliest_conflict():
    from src.mapf.lns_sat import (
        find_earliest_conflict,
        repair_earliest_conflict,
    )

    grid = [
        [0, 0, 0],
        [0, 0, 0],
    ]

    paths = {
        0: [
            (0, 0),
            (1, 0),
            (2, 0),
            (2, 0),
            (2, 0),
        ],
        1: [
            (2, 0),
            (1, 0),
            (0, 0),
            (0, 0),
            (0, 0),
        ],
    }

    result = repair_earliest_conflict(
        grid=grid,
        paths=paths,
        radius=1,
        max_expansions=3,
    )

    assert result is not None
    assert result["repaired"] is True

    remaining_conflict = find_earliest_conflict(
        result["paths"]
    )

    assert remaining_conflict is None
def test_repair_until_conflict_free():
    from src.mapf.lns_sat import (
        find_earliest_conflict,
        repair_until_conflict_free,
    )

    grid = [
        [0, 0, 0],
        [0, 0, 0],
    ]

    paths = {
        0: [
            (0, 0),
            (1, 0),
            (2, 0),
            (2, 0),
            (2, 0),
        ],
        1: [
            (2, 0),
            (1, 0),
            (0, 0),
            (0, 0),
            (0, 0),
        ],
    }

    result = repair_until_conflict_free(
        grid=grid,
        paths=paths,
        radius=1,
        max_expansions=3,
        max_iterations=10,
    )

    assert result is not None
    assert find_earliest_conflict(
        result["paths"]
    ) is None
def test_solve_lns_sat_two_agents():
    from src.mapf.lns_sat import (
        find_earliest_conflict,
        solve_lns_sat,
    )

    grid = [
        [0, 0, 0],
        [0, 0, 0],
    ]

    result = solve_lns_sat(
        grid=grid,
        starts={
            0: (0, 0),
            1: (2, 0),
        },
        goals={
            0: (2, 0),
            1: (0, 0),
        },
        radius=1,
        max_expansions=3,
        max_iterations=10,
    )

    assert result is not None
    assert find_earliest_conflict(
        result["paths"]
    ) is None
def test_extend_paths_to_makespan():
    from src.mapf.lns_sat import (
        extend_paths_to_makespan,
    )

    paths = {
        0: [
            (0, 0),
            (1, 0),
            (2, 0),
        ],
    }

    extended = extend_paths_to_makespan(
        paths=paths,
        makespan=4,
    )

    assert len(extended[0]) == 5
    assert extended[0][-1] == (2, 0)
