from src.marl.observation_builder import build_agent_input


def test_m2_and_m3_have_same_shape_but_different_information():
    grid = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]

    m2 = build_agent_input(
        method="M2",
        grid=grid,
        center_position=(1, 1),
        goal_position=(2, 2),
        trusted_positions=[(2, 1)],
        reachable_occupancies={(0, 1)},
        aoi=3,
        reachable_size=4,
    )

    m3 = build_agent_input(
        method="M3",
        grid=grid,
        center_position=(1, 1),
        goal_position=(2, 2),
        trusted_positions=[(2, 1)],
        reachable_occupancies={(0, 1)},
        aoi=3,
        reachable_size=4,
    )

    assert m2.shape == (247,)
    assert m3.shape == (247,)
    assert not (m2 == m3).all()
