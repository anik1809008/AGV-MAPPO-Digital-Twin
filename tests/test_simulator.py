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
def test_goal_tracking():
    grid = [[0, 0, 0]]

    sim = GroundTruthSimulator(
        grid=grid,
        agent_positions={0: (0, 0)},
        agent_goals={0: (2, 0)},
    )

    assert sim.is_at_goal(0) is False
    assert sim.all_goals_reached() is False

    sim.move_agent(0, Action.EAST)
    sim.move_agent(0, Action.EAST)

    assert sim.is_at_goal(0) is True
    assert sim.all_goals_reached() is True
def test_deadlock_after_ten_non_progress_steps():
    grid = [[0, 0, 0]]

    sim = GroundTruthSimulator(
        grid=grid,
        agent_positions={0: (0, 0)},
        agent_goals={0: (2, 0)},
        deadlock_threshold=10,
    )

    for _ in range(10):
        previous_positions = dict(sim.agent_positions)
        sim.update_progress(previous_positions)

    assert sim.non_progress_steps == 10
    assert sim.is_deadlocked() is True
