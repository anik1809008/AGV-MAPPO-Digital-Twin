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
