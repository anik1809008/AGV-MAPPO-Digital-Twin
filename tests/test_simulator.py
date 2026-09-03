from src.environment.actions import Action
from src.environment.simulator import GroundTruthSimulator


def make_simulator():
    grid = [
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ]

    return GroundTruthSimulator(
        grid=grid,
        agent_positions={0: (0, 0)},
    )


def test_valid_movement():
    sim = make_simulator()

    result = sim.move_agent(0, Action.EAST)

    assert result == (1, 0)


def test_wall_blocks_movement():
    sim = make_simulator()

    sim.move_agent(0, Action.EAST)
    result = sim.move_agent(0, Action.SOUTH)

    assert result == (1, 0)


def test_boundary_blocks_movement():
    sim = make_simulator()

    result = sim.move_agent(0, Action.WEST)

    assert result == (0, 0)


def test_wait_keeps_position():
    sim = make_simulator()

    result = sim.move_agent(0, Action.WAIT)

    assert result == (0, 0)
def test_vertex_conflict_blocks_joint_move():
    grid = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]

    sim = GroundTruthSimulator(
        grid=grid,
        agent_positions={
            0: (0, 1),
            1: (2, 1),
        },
    )

    result = sim.step_joint({
        0: Action.EAST,
        1: Action.WEST,
    })

    assert result["success"] is False
    assert result["vertex_conflicts"] == {
        (1, 1): [0, 1]
    }

    assert sim.agent_positions == {
        0: (0, 1),
        1: (2, 1),
    }
def test_edge_swap_conflict_blocks_joint_move():
    grid = [[0, 0]]

    sim = GroundTruthSimulator(
        grid=grid,
        agent_positions={
            0: (0, 0),
            1: (1, 0),
        },
    )

    result = sim.step_joint({
        0: Action.EAST,
        1: Action.WEST,
    })

    assert result["success"] is False
    assert result["vertex_conflicts"] == {}
    assert result["edge_swap_conflicts"] == [(0, 1)]

    assert sim.agent_positions == {
        0: (0, 0),
        1: (1, 0),
    }
def test_successful_joint_move():
    grid = [
        [0, 0, 0],
        [0, 0, 0],
    ]

    sim = GroundTruthSimulator(
        grid=grid,
        agent_positions={
            0: (0, 0),
            1: (2, 1),
        },
    )

    result = sim.step_joint({
        0: Action.EAST,
        1: Action.WEST,
    })

    assert result["success"] is True
    assert result["vertex_conflicts"] == {}
    assert result["edge_swap_conflicts"] == []

    assert sim.agent_positions == {
        0: (1, 0),
        1: (1, 1),
    }
