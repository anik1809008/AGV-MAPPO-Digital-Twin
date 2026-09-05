from src.marl.environment_adapter import build_all_agent_inputs


def test_build_all_agent_inputs():
    grid = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]

    agent_positions = [
        (0, 0),
        (2, 2),
    ]

    agent_goals = [
        (2, 2),
        (0, 0),
    ]

    trusted_positions = {
        0: (0, 0),
        1: (2, 2),
    }

    reachable_occupancies = {
        0: {(0, 0)},
        1: {(2, 2)},
    }

    aoi_values = {
        0: 0,
        1: 0,
    }

    observations = build_all_agent_inputs(
        method="M3",
        grid=grid,
        agent_positions=agent_positions,
        agent_goals=agent_goals,
        trusted_positions=trusted_positions,
        reachable_occupancies=reachable_occupancies,
        aoi_values=aoi_values,
    )

    assert len(observations) == 2
    assert observations[0].shape == (247,)
    assert observations[1].shape == (247,)
