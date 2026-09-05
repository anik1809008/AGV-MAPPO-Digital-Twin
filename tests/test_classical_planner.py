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
from src.baselines.multi_agent_classical import has_joint_conflict


def test_multi_agent_vertex_conflict():
    assert has_joint_conflict(
        current_positions={
            0: (0, 0),
            1: (2, 0),
        },
        next_positions={
            0: (1, 0),
            1: (1, 0),
        },
    ) is True


def test_multi_agent_edge_swap_conflict():
    assert has_joint_conflict(
        current_positions={
            0: (0, 0),
            1: (1, 0),
        },
        next_positions={
            0: (1, 0),
            1: (0, 0),
        },
    ) is True
