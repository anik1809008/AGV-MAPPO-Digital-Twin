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
from src.marl.observations import build_scalar_features


def test_scalar_features():
    features = build_scalar_features(
        center_position=(8, 5),
        goal_position=(3, 9),
        aoi=2,
        reachable_size=3,
    )

    assert features.shape == (4,)
    assert features.tolist() == [-5.0, 4.0, 2.0, 3.0]
from src.marl.observations import flatten_mappo_input


def test_flatten_mappo_input():
    grid = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]

    spatial = build_local_observation(
        grid=grid,
        center_position=(1, 1),
        trusted_positions=[],
        reachable_occupancies=set(),
        window_size=9,
    )

    scalars = build_scalar_features(
        center_position=(1, 1),
        goal_position=(2, 2),
        aoi=1,
        reachable_size=2,
    )

    vector = flatten_mappo_input(
        spatial,
        scalars,
    )

    assert vector.shape == (247,)
    assert str(vector.dtype) == "float32"
