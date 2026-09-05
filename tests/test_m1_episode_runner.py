from src.environment.simulator import GroundTruthSimulator
from src.baselines.m1_episode_runner import run_m1_episode


def test_run_m1_episode_reaches_goals():
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

    result = run_m1_episode(
        simulator=simulator,
        max_steps=5,
    )

    assert result["planning_failed"] is False
    assert result["collision"] is False
    assert result["all_goals_reached"] is True
    assert result["steps"] == 2
