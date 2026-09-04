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
def test_reachable_occupancy_conflict_detection():
    shield = SafetyShield([
        [0, 0, 0],
        [0, 0, 0],
    ])

    reachable = {
        1: {(1, 0), (2, 0)},
        2: {(2, 1)},
    }

    assert shield.intersects_reachable_occupancy(
        agent_id=0,
        candidate_position=(1, 0),
        reachable_occupancies=reachable,
    ) is True

    assert shield.intersects_reachable_occupancy(
        agent_id=0,
        candidate_position=(0, 1),
        reachable_occupancies=reachable,
    ) is False
def test_possible_next_positions_from_uncertain_state():
    shield = SafetyShield([
        [0, 0, 0],
        [0, 0, 0],
    ])

    possible_current = {
        (0, 0),
        (1, 0),
    }

    next_positions = shield.possible_next_positions(
        possible_current_positions=possible_current,
        action=Action.EAST,
    )

    assert next_positions == {
        (1, 0),
        (2, 0),
    }

def test_combined_action_safety_check():
    shield = SafetyShield([
        [0, 0, 0],
        [0, 0, 0],
    ])

    possible_current = {(0, 0)}

    other_current = {
        1: (2, 0),
    }

    other_next = {
        1: (2, 0),
    }

    reachable = {
        1: {(1, 0), (2, 0)},
    }

    assert shield.is_action_safe(
        agent_id=0,
        possible_current_positions=possible_current,
        action=Action.EAST,
        other_current_positions=other_current,
        other_next_positions=other_next,
        reachable_occupancies=reachable,
    ) is False

    assert shield.is_action_safe(
        agent_id=0,
        possible_current_positions=possible_current,
        action=Action.SOUTH,
        other_current_positions=other_current,
        other_next_positions=other_next,
        reachable_occupancies=reachable,
    ) is True
