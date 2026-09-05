from src.baselines.classical_planner import shortest_path


def test_shortest_path_avoids_obstacle():
    grid = [
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ]

    path = shortest_path(
        grid=grid,
        start=(0, 0),
        goal=(2, 2),
    )

    assert path is not None
    assert len(path) == 4
