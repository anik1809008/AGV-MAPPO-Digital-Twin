from unittest.mock import Mock

import pytest

from src.marl.dt_decision_support import (
    get_dt_decision_positions,
    update_dt_progress,
    select_dt_recovery_action,
    reserve_dt_next_positions,
)
from src.safety.shield import SafetyShield


def test_decisions_use_trusted_position():
    result = get_dt_decision_positions(
        trusted_positions={0: (1, 0)},
        possible_positions={0: {(1, 0), (2, 0)}},
        agent_ids=[0],
    )
    assert result[0] == (1, 0)


def test_missing_uncertainty_is_rejected():
    with pytest.raises(ValueError):
        get_dt_decision_positions(
            {0: (1, 0)}, {}, [0]
        )


def test_stale_telemetry_does_not_update_progress():
    monitor = Mock()
    monitor.is_stagnating.return_value = False

    update_dt_progress(
        monitor, 0, (1, 0), (3, 0), aoi=2
    )

    monitor.update.assert_not_called()


def test_fresh_telemetry_updates_progress():
    monitor = Mock()

    update_dt_progress(
        monitor, 0, (1, 0), (3, 0), aoi=0
    )

    monitor.update.assert_called_once()


def test_ambiguous_position_blocks_recovery():
    planner = Mock()

    result = select_dt_recovery_action(
        planner=planner,
        agent_id=0,
        possible_positions={0: {(0, 0), (1, 0)}},
        reachable_occupancies={0: {(0, 0), (1, 0)}},
        goal=(3, 0),
    )

    assert result is None
    planner.select_recovery_action.assert_not_called()


def test_recovery_considers_other_agent_uncertainty():
    planner = Mock(return_value=3)

    select_dt_recovery_action(
        planner=planner,
        agent_id=0,
        possible_positions={0: {(0, 0)}},
        reachable_occupancies={
            0: {(0, 0)},
            1: {(2, 0), (3, 0)},
        },
        goal=(4, 0),
    )

    assert planner.select_recovery_action.call_args.kwargs[
        "occupied_positions"
    ] == {(2, 0), (3, 0)}


def test_reservation_includes_all_possible_positions():
    shield = SafetyShield([[0] * 5])

    reserved = reserve_dt_next_positions(
        shield=shield,
        possible_positions={(0, 0), (1, 0)},
        action=3,
    )

    assert reserved == {(0, 0), (1, 0), (2, 0)}
