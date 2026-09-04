from src.environment.actions import Action
from src.safety.shield import SafetyShield


def make_shield():
    grid = [
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ]

    return SafetyShield(grid)


def test_valid_move():
    shield = make_shield()

    assert shield.next_position(
        (0, 0),
        Action.EAST,
    ) == (1, 0)


def test_wall_blocks_move():
    shield = make_shield()

    assert shield.next_position(
        (1, 0),
        Action.SOUTH,
    ) == (1, 0)


def test_boundary_blocks_move():
    shield = make_shield()

    assert shield.next_position(
        (0, 0),
        Action.WEST,
    ) == (0, 0)
def test_vertex_conflict_detection():
    shield = SafetyShield([
        [0, 0, 0],
        [0, 0, 0],
    ])

    other_next_positions = {
        1: (1, 0),
        2: (2, 1),
    }

    assert shield.has_vertex_conflict(
        agent_id=0,
        candidate_position=(1, 0),
        other_next_positions=other_next_positions,
    ) is True

    assert shield.has_vertex_conflict(
        agent_id=0,
        candidate_position=(0, 1),
        other_next_positions=other_next_positions,
    ) is False
def test_edge_swap_conflict_detection():
    shield = SafetyShield([[0, 0]])

    current_positions = {
        0: (0, 0),
        1: (1, 0),
    }

    next_positions = {
        0: (1, 0),
        1: (0, 0),
    }

    assert shield.has_edge_swap_conflict(
        agent_id=0,
        current_position=(0, 0),
        candidate_position=(1, 0),
        other_current_positions=current_positions,
        other_next_positions=next_positions,
    ) is True
