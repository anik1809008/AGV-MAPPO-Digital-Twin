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
def test_m6_uses_aoi_without_reachable_occupancy():
    grid = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]

    vector = build_agent_input(
        method="M6",
        grid=grid,
        center_position=(1, 1),
        goal_position=(2, 1),
        trusted_positions=[(1, 1)],
        reachable_occupancies={
            (1, 1),
            (0, 1),
        },
        aoi=3,
        reachable_size=2,
        window_size=3,
    )

    # Last four values:
    # dx, dy, AoI, reachable_size
    assert vector[-4] == 0.5
    assert vector[-3] == 0.0
    assert vector[-2] == 3.0
    assert vector[-1] == 1.0

    # Reachable-occupancy channel must remain empty.
    spatial = vector[:-4].reshape(3, 3, 3)
    assert spatial[2].sum() == 0.0
