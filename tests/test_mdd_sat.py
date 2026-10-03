from src.mapf.mdd_sat import shortest_distances


def test_shortest_distances_on_open_grid():
    grid = [
        [0, 0, 0],
        [0, 0, 0],
    ]

    distances = shortest_distances(
        grid=grid,
        goal=(2, 0),
    )

    assert distances[(2, 0)] == 0
    assert distances[(1, 0)] == 1
    assert distances[(0, 0)] == 2
    assert distances[(0, 1)] == 3


def test_build_mdd_contains_start_and_goal():
    from src.mapf.mdd_sat import build_mdd

    grid = [
        [0, 0, 0],
        [0, 0, 0],
    ]

    mdd = build_mdd(
        grid=grid,
        start=(0, 0),
        goal=(2, 0),
        horizon=2,
    )

    assert mdd[0] == {(0, 0)}
    assert (2, 0) in mdd[2]
def test_create_node_variables_are_unique():
    from src.mapf.mdd_sat import create_node_variables

    mdds = {
        0: [
            {(0, 0)},
            {(1, 0)},
        ],
        1: [
            {(2, 0)},
            {(1, 0)},
        ],
    }

    variables = create_node_variables(mdds)

    assert len(variables) == 4
    assert len(set(variables.values())) == 4
    assert min(variables.values()) == 1
def test_exactly_one_position_constraints():
    from pysat.formula import CNF

    from src.mapf.mdd_sat import (
        add_exactly_one_position_constraints,
        create_node_variables,
    )

    mdds = {
        0: [
            {(0, 0)},
            {(0, 0), (1, 0)},
        ],
    }

    variables = create_node_variables(mdds)
    cnf = CNF()

    add_exactly_one_position_constraints(
        cnf=cnf,
        mdds=mdds,
        variables=variables,
    )

    assert len(cnf.clauses) == 3
def test_movement_constraints_allow_valid_transition():
    from pysat.formula import CNF

    from src.mapf.mdd_sat import (
        add_movement_constraints,
        create_node_variables,
    )

    mdds = {
        0: [
            {(0, 0)},
            {(0, 0), (1, 0)},
        ],
    }

    variables = create_node_variables(mdds)
    cnf = CNF()

    add_movement_constraints(
        cnf=cnf,
        mdds=mdds,
        variables=variables,
    )

    assert len(cnf.clauses) == 1
    assert len(cnf.clauses[0]) == 3
def test_vertex_conflict_constraints():
    from pysat.formula import CNF

    from src.mapf.mdd_sat import (
        add_vertex_conflict_constraints,
        create_node_variables,
    )

    mdds = {
        0: [
            {(0, 0)},
            {(1, 0)},
        ],
        1: [
            {(2, 0)},
            {(1, 0)},
        ],
    }

    variables = create_node_variables(mdds)
    cnf = CNF()

    add_vertex_conflict_constraints(
        cnf=cnf,
        mdds=mdds,
        variables=variables,
    )

    assert len(cnf.clauses) == 1
def test_edge_swap_constraints():
    from pysat.formula import CNF

    from src.mapf.mdd_sat import (
        add_edge_swap_constraints,
        create_node_variables,
    )

    mdds = {
        0: [
            {(0, 0)},
            {(1, 0)},
        ],
        1: [
            {(1, 0)},
            {(0, 0)},
        ],
    }

    variables = create_node_variables(mdds)
    cnf = CNF()

    add_edge_swap_constraints(
        cnf=cnf,
        mdds=mdds,
        variables=variables,
    )

    assert len(cnf.clauses) == 1
def test_solve_fixed_makespan_single_agent():
    from src.mapf.mdd_sat import solve_fixed_makespan

    grid = [
        [0, 0, 0],
        [0, 0, 0],
    ]

    paths = solve_fixed_makespan(
        grid=grid,
        starts={
            0: (0, 0),
        },
        goals={
            0: (2, 0),
        },
        makespan=2,
    )

    assert paths is not None
    assert paths[0][0] == (0, 0)
    assert paths[0][-1] == (2, 0)
    assert len(paths[0]) == 3
def test_solve_fixed_makespan_two_agents():
    from src.mapf.mdd_sat import solve_fixed_makespan

    grid = [
        [0, 0, 0],
        [0, 0, 0],
    ]

    paths = solve_fixed_makespan(
        grid=grid,
        starts={
            0: (0, 0),
            1: (2, 0),
        },
        goals={
            0: (2, 0),
            1: (0, 0),
        },
        makespan=4,
    )

    assert paths is not None

    for timestep in range(5):
        assert paths[0][timestep] != paths[1][timestep]

    for timestep in range(4):
        assert not (
            paths[0][timestep] == paths[1][timestep + 1]
            and paths[1][timestep] == paths[0][timestep + 1]
        )
def test_solve_minimum_makespan_two_agents():
    from src.mapf.mdd_sat import solve_minimum_makespan

    grid = [
        [0, 0, 0],
        [0, 0, 0],
    ]

    result = solve_minimum_makespan(
        grid=grid,
        starts={
            0: (0, 0),
            1: (2, 0),
        },
        goals={
            0: (2, 0),
            1: (0, 0),
        },
        max_makespan=6,
    )

    assert result is not None
    assert result["makespan"] == 4
def test_minimum_makespan_single_agent_is_shortest_path_length():
    from src.mapf.mdd_sat import solve_minimum_makespan

    grid = [
        [0, 0, 0, 0],
    ]

    result = solve_minimum_makespan(
        grid=grid,
        starts={
            0: (0, 0),
        },
        goals={
            0: (3, 0),
        },
        max_makespan=5,
    )

    assert result is not None
    assert result["makespan"] == 3
