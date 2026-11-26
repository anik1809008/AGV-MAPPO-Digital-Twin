
"""Tests for DT-inferred completion eligibility."""

import pytest

from src.hybrid.dt_inferred_completion import (
    get_inferred_completion_positions,
)


def example():
    return {
        "dt": {
            "possible_current_positions": {
                0: {(0, 0)},
                1: {(2, 0)},
                2: {(4, 0)},
            },
            "aoi_values": {
                0: 2,
                1: 2,
                2: 2,
            },
        },
        "goals": {
            0: (1, 0),
            1: (2, 0),
            2: (4, 0),
        },
        "immediate_probability": 1.0,
        "pending_commands": False,
        "handover_active": False,
    }


def test_accept_one_unfinished_agent():
    args = example()

    assert get_inferred_completion_positions(**args) == {
        0: (0, 0),
        1: (2, 0),
        2: (4, 0),
    }


def test_accept_two_unfinished_agents():
    args = example()
    args["goals"][1] = (3, 0)

    assert get_inferred_completion_positions(
        **args
    ) is not None


def test_reject_three_unfinished_agents():
    args = example()
    args["goals"][1] = (3, 0)
    args["goals"][2] = (5, 0)

    assert get_inferred_completion_positions(
        **args
    ) is None


def test_reject_stochastic_execution():
    args = example()
    args["immediate_probability"] = 0.8

    assert get_inferred_completion_positions(
        **args
    ) is None


def test_reject_pending_commands():
    args = example()
    args["pending_commands"] = True

    assert get_inferred_completion_positions(
        **args
    ) is None


def test_reject_active_handover():
    args = example()
    args["handover_active"] = True

    assert get_inferred_completion_positions(
        **args
    ) is None


def test_reject_ambiguous_positions():
    args = example()

    args["dt"]["possible_current_positions"][0] = {
        (0, 0), (1, 0)
    }

    assert get_inferred_completion_positions(
        **args
    ) is None


def test_reject_excessive_aoi():
    args = example()
    args["dt"]["aoi_values"][0] = 3

    assert get_inferred_completion_positions(
        **args
    ) is None


def test_preserve_original_fresh_state_path():
    args = example()

    args["dt"]["aoi_values"] = {
        0: 0, 1: 0, 2: 0
    }

    assert get_inferred_completion_positions(
        **args
    ) is None


def test_reject_missing_agent():
    args = example()

    del args["dt"]["possible_current_positions"][2]

    assert get_inferred_completion_positions(
        **args
    ) is None


def test_reject_duplicate_positions():
    args = example()

    args["dt"]["possible_current_positions"][1] = {
        (0, 0)
    }

    assert get_inferred_completion_positions(
        **args
    ) is None


def test_reject_all_agents_finished():
    args = example()
    args["goals"][0] = (0, 0)

    assert get_inferred_completion_positions(
        **args
    ) is None


def test_reject_negative_aoi():
    args = example()
    args["dt"]["aoi_values"][0] = -1

    assert get_inferred_completion_positions(
        **args
    ) is None


def test_reject_missing_aoi():
    args = example()

    del args["dt"]["aoi_values"][0]

    assert get_inferred_completion_positions(
        **args
    ) is None


def test_invalid_maximum_aoi():
    args = example()

    with pytest.raises(ValueError):
        get_inferred_completion_positions(
            **args,
            max_aoi=0,
        )
