"""Tests for experimental DT-inferred recovery route validation."""

import pytest

from src.hybrid.dt_inferred_completion import (
    inferred_recovery_route_is_valid,
)


def example():
    return {
        "plan": {
            "recovery_expected_positions": [
                {0: (0, 0), 1: (3, 0)},
                {0: (1, 0), 1: (3, 0)},
                {0: (2, 0), 1: (3, 0)},
            ],
        },
        "dt": {
            "possible_current_positions": {
                0: {(1, 0)},
                1: {(3, 0)},
            },
            "aoi_values": {
                0: 2,
                1: 2,
            },
        },
        "index": 1,
        "immediate_probability": 1.0,
        "pending_commands": False,
        "handover_active": False,
    }


def test_accept_valid_inferred_route():
    assert inferred_recovery_route_is_valid(**example())


def test_accept_fresh_telemetry_during_recovery():
    args = example()
    args["dt"]["aoi_values"] = {0: 0, 1: 0}

    assert inferred_recovery_route_is_valid(**args)


def test_reject_stochastic_execution():
    args = example()
    args["immediate_probability"] = 0.8

    assert not inferred_recovery_route_is_valid(**args)


def test_reject_pending_commands():
    args = example()
    args["pending_commands"] = True

    assert not inferred_recovery_route_is_valid(**args)


def test_reject_active_handover():
    args = example()
    args["handover_active"] = True

    assert not inferred_recovery_route_is_valid(**args)


def test_reject_excessive_aoi():
    args = example()
    args["dt"]["aoi_values"][0] = 3

    assert not inferred_recovery_route_is_valid(**args)


def test_reject_negative_aoi():
    args = example()
    args["dt"]["aoi_values"][0] = -1

    assert not inferred_recovery_route_is_valid(**args)


def test_reject_ambiguous_position():
    args = example()
    args["dt"]["possible_current_positions"][0] = {
        (1, 0), (2, 0)
    }

    assert not inferred_recovery_route_is_valid(**args)


def test_reject_route_divergence():
    args = example()
    args["dt"]["possible_current_positions"][0] = {(0, 0)}

    assert not inferred_recovery_route_is_valid(**args)


def test_reject_missing_agent():
    args = example()
    del args["dt"]["possible_current_positions"][1]

    assert not inferred_recovery_route_is_valid(**args)


def test_reject_missing_aoi():
    args = example()
    del args["dt"]["aoi_values"][1]

    assert not inferred_recovery_route_is_valid(**args)


def test_reject_negative_index():
    args = example()
    args["index"] = -1

    assert not inferred_recovery_route_is_valid(**args)


def test_reject_expired_route():
    args = example()
    args["index"] = 3

    assert not inferred_recovery_route_is_valid(**args)


def test_reject_missing_expected_positions():
    args = example()
    args["plan"] = {}

    assert not inferred_recovery_route_is_valid(**args)


def test_reject_duplicate_occupancy():
    args = example()

    args["plan"]["recovery_expected_positions"][1] = {
        0: (1, 0),
        1: (1, 0),
    }

    args["dt"]["possible_current_positions"] = {
        0: {(1, 0)},
        1: {(1, 0)},
    }

    assert not inferred_recovery_route_is_valid(**args)


def test_invalid_maximum_aoi():
    with pytest.raises(ValueError):
        inferred_recovery_route_is_valid(
            **example(),
            max_aoi=0,
        )


def test_reject_empty_expected_state():
    args = example()

    args["plan"]["recovery_expected_positions"][1] = {}
    args["dt"]["possible_current_positions"] = {}
    args["dt"]["aoi_values"] = {}

    assert not inferred_recovery_route_is_valid(**args)


def test_accept_final_expected_position():
    args = example()
    args["index"] = 2
    args["dt"]["possible_current_positions"][0] = {(2, 0)}

    assert inferred_recovery_route_is_valid(**args)
