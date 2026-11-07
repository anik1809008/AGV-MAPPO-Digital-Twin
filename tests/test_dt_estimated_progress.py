from unittest.mock import Mock

import pytest

from src.marl.dt_decision_support import update_dt_estimated_progress


def test_singleton_updates_monitor():
    monitor = Mock()
    monitor.update.return_value = True

    result = update_dt_estimated_progress(
        monitor=monitor,
        agent_id=0,
        possible_positions={(4, 2)},
        goal=(8, 2),
    )

    assert result is True
    monitor.update.assert_called_once_with(
        agent_id=0,
        position=(4, 2),
        goal=(8, 2),
    )


def test_ambiguous_estimate_does_not_update():
    monitor = Mock()
    monitor.is_stagnating.return_value = False

    result = update_dt_estimated_progress(
        monitor=monitor,
        agent_id=0,
        possible_positions={(4, 2), (5, 2)},
        goal=(8, 2),
    )

    assert result is False
    monitor.update.assert_not_called()
    monitor.is_stagnating.assert_called_once_with(0)


def test_empty_estimate_rejected():
    with pytest.raises(ValueError, match="Missing DT position"):
        update_dt_estimated_progress(
            monitor=Mock(),
            agent_id=0,
            possible_positions=set(),
            goal=(8, 2),
        )


def test_agent_at_goal_never_reports_stagnation():
    from src.safety.progress_monitor import ProgressMonitor

    monitor = ProgressMonitor(
        grid=[[0] * 10 for _ in range(10)],
        stagnation_steps=3,
    )

    for _ in range(20):
        result = update_dt_estimated_progress(
            monitor=monitor,
            agent_id=0,
            possible_positions={(5, 5)},
            goal=(5, 5),
        )
        assert result is False
        assert monitor.is_stagnating(0) is False
