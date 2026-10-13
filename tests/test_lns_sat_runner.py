from src.environment.simulator import GroundTruthSimulator
from src.mapf.lns_sat_runner import run_lns_sat_episode


def test_run_lns_sat_episode_reaches_goals():
    grid = [
        [0, 0, 0],
        [0, 0, 0],
    ]

    simulator = GroundTruthSimulator(
        grid=grid,
        agent_positions={
            0: (0, 0),
            1: (2, 0),
        },
        agent_goals={
            0: (2, 0),
            1: (0, 0),
        },
    )

    result = run_lns_sat_episode(
        simulator=simulator,
    )

    assert result["planning_failed"] is False
    assert result["collision"] is False
    assert result["all_goals_reached"] is True
