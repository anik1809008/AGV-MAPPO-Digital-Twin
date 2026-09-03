from src.marl.observations import build_local_observation


def test_local_observation_shape():
    grid = [
        [0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]

    obs = build_local_observation(
        grid=grid,
        center_position=(2, 2),
        trusted_positions=[(3, 2)],
        reachable_occupancies={(1, 2), (1, 1)},
        window_size=9,
    )

    assert obs.shape == (3, 9, 9)
    assert int(obs[1].sum()) == 1
    assert int(obs[2].sum()) == 2


def test_even_window_size_rejected():
    grid = [[0]]

    try:
        build_local_observation(
            grid=grid,
            center_position=(0, 0),
            trusted_positions=[],
            window_size=8,
        )
    except ValueError:
        return

    assert False, "Expected ValueError for even window size"
from src.marl.observations import build_m2_observation, build_m3_observation


def test_m2_vs_m3_reliability_channel():
    grid = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]

    m2 = build_m2_observation(
        grid=grid,
        center_position=(1, 1),
        trusted_positions=[(2, 1)],
    )

    m3 = build_m3_observation(
        grid=grid,
        center_position=(1, 1),
        trusted_positions=[(2, 1)],
        reachable_occupancies={(0, 1), (0, 0)},
    )

    assert int(m2[2].sum()) == 0
    assert int(m3[2].sum()) == 2
