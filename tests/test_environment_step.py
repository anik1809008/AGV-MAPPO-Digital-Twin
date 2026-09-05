from src.environment.actions import Action
from src.environment.simulator import GroundTruthSimulator
from src.marl.environment_step import execute_training_step


def test_execute_training_step():
    grid = [
        [0, 0, 0],
        [0, 0, 0],
    ]

    simulator = GroundTruthSimulator(
        grid=grid,
        agent_positions={
            0: (0, 0),
            1: (2, 1),
        },
        agent_goals={
            0: (2, 0),
            1: (0, 1),
        },
    )

    result = execute_training_step(
        simulator=simulator,
        actions={
            0: Action.EAST,
            1: Action.WEST,
        },
    )

    assert result["previous_positions"] == [
        (0, 0),
        (2, 1),
    ]

    assert result["current_positions"] == [
        (1, 0),
        (1, 1),
    ]

    assert result["rewards"] == [
        0.08,
        0.08,
    ]

    assert result["collision"] is False
    assert result["dones"] == [
        False,
        False,
    ]
