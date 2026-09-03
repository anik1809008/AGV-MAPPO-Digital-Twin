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
