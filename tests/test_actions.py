from src.environment.actions import Action, ACTION_DELTAS


def test_action_values():
    assert Action.WAIT == 0
    assert Action.NORTH == 1
    assert Action.SOUTH == 2
    assert Action.EAST == 3
    assert Action.WEST == 4


def test_action_deltas():
    assert ACTION_DELTAS[Action.WAIT] == (0, 0)
    assert ACTION_DELTAS[Action.NORTH] == (0, -1)
    assert ACTION_DELTAS[Action.SOUTH] == (0, 1)
    assert ACTION_DELTAS[Action.EAST] == (1, 0)
    assert ACTION_DELTAS[Action.WEST] == (-1, 0)
