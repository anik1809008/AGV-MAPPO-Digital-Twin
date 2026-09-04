from src.baselines.stale_wait import ReachableSetWaitBaseline
from src.environment.actions import Action


def test_action_allowed_below_threshold():
    baseline = ReachableSetWaitBaseline(threshold=3)

    action = baseline.select_action(
        normal_action=Action.EAST,
        reachable_size=2,
    )

    assert action == Action.EAST


def test_wait_at_threshold():
    baseline = ReachableSetWaitBaseline(threshold=3)

    action = baseline.select_action(
        normal_action=Action.EAST,
        reachable_size=3,
    )

    assert action == Action.WAIT


def test_wait_above_threshold():
    baseline = ReachableSetWaitBaseline(threshold=3)

    action = baseline.select_action(
        normal_action=Action.WEST,
        reachable_size=4,
    )

    assert action == Action.WAIT
