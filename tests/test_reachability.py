from src.digital_twin.reachability import compute_reachable_occupancy
from src.environment.actions import Action


def test_two_west_commands_expand_reachable_set():
    grid = [[0, 0, 0, 0, 0]]

    reachable = compute_reachable_occupancy(
        grid=grid,
        last_trusted_position=(3, 0),
        command_history=[
            Action.WEST,
            Action.WEST,
        ],
    )

    assert reachable == {
        (3, 0),
        (2, 0),
        (1, 0),
    }


def test_reachability_follows_command_direction():
    grid = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]

    reachable = compute_reachable_occupancy(
        grid=grid,
        last_trusted_position=(1, 1),
        command_history=[
            Action.WEST,
            Action.NORTH,
        ],
    )

    assert (0, 1) in reachable
    assert (0, 0) in reachable
    assert (1, 1) in reachable


def test_wall_blocks_reachable_motion():
    grid = [[0, 1, 0]]

    reachable = compute_reachable_occupancy(
        grid=grid,
        last_trusted_position=(0, 0),
        command_history=[Action.EAST],
    )

    assert reachable == {(0, 0)}
def test_reachability_preserves_fifo_command_order():
    grid = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]

    reachable = compute_reachable_occupancy(
        grid=grid,
        last_trusted_position=(1, 1),
        command_history=[
            Action.WEST,
            Action.NORTH,
        ],
    )

    assert reachable == {
        (1, 1),
        (0, 1),
        (0, 0),
    }

    assert (1, 0) not in reachable
from src.digital_twin.reachability import (
    compute_possible_transitions,
)
from src.environment.actions import Action


def test_possible_transitions_follow_command_prefixes():
    grid = [
        [0, 0, 0],
    ]

    transitions = compute_possible_transitions(
        grid=grid,
        last_trusted_position=(0, 0),
        command_history=[
            (1, Action.EAST),
            (2, Action.EAST),
        ],
    )

    assert transitions == {
        ((0, 0), (0, 0)),
        ((0, 0), (1, 0)),
        ((1, 0), (1, 0)),
        ((1, 0), (2, 0)),
        ((2, 0), (2, 0)),
    }
def test_reachability_stays_locked_after_goal():
    grid = [
        [0, 0, 0, 0],
    ]

    reachable = compute_reachable_occupancy(
        grid=grid,
        last_trusted_position=(1, 0),
        command_history=[
            (1, Action.EAST),
            (2, Action.EAST),
        ],
        goal=(2, 0),
    )

    assert reachable == {
        (1, 0),
        (2, 0),
    }


def test_possible_transitions_stay_locked_after_goal():
    grid = [
        [0, 0, 0, 0],
    ]

    transitions = compute_possible_transitions(
        grid=grid,
        last_trusted_position=(1, 0),
        command_history=[
            (1, Action.EAST),
            (2, Action.EAST),
        ],
        goal=(2, 0),
    )

    assert transitions == {
        ((1, 0), (1, 0)),
        ((1, 0), (2, 0)),
        ((2, 0), (2, 0)),
    }
