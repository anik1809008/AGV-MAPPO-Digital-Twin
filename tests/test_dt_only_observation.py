import numpy as np
import pytest

from src.marl.dt_only_observation import build_dt_only_agent_inputs
from src.marl.environment_adapter import build_all_agent_inputs


def inputs():
    return {
        "grid": [[0] * 7 for _ in range(7)],
        "agent_goals": {0: (6, 6), 1: (0, 6)},
        "trusted_positions": {0: (1, 1), 1: (5, 1)},
        "reachable_occupancies": {
            0: {(1, 1), (2, 1)},
            1: {(5, 1)},
        },
        "aoi_values": {0: 1, 1: 0},
    }


def test_dt_only_observations_match_trusted_centers():
    data = inputs()

    actual = build_dt_only_agent_inputs(**data)

    expected = build_all_agent_inputs(
        method="M7",
        grid=data["grid"],
        agent_positions=[(1, 1), (5, 1)],
        agent_goals=[(6, 6), (0, 6)],
        trusted_positions=data["trusted_positions"],
        reachable_occupancies=data["reachable_occupancies"],
        aoi_values=data["aoi_values"],
    )

    assert len(actual) == 2

    for result, reference in zip(actual, expected):
        np.testing.assert_array_equal(result, reference)


def test_legacy_observation_depends_on_simulator_position():
    data = inputs()

    original = build_all_agent_inputs(
        method="M7",
        grid=data["grid"],
        agent_positions=[(1, 1), (5, 1)],
        agent_goals=[(6, 6), (0, 6)],
        trusted_positions=data["trusted_positions"],
        reachable_occupancies=data["reachable_occupancies"],
        aoi_values=data["aoi_values"],
    )

    changed = build_all_agent_inputs(
        method="M7",
        grid=data["grid"],
        agent_positions=[(2, 1), (5, 1)],
        agent_goals=[(6, 6), (0, 6)],
        trusted_positions=data["trusted_positions"],
        reachable_occupancies=data["reachable_occupancies"],
        aoi_values=data["aoi_values"],
    )

    assert not np.array_equal(original[0], changed[0])


def test_missing_trusted_position_is_rejected():
    data = inputs()
    del data["trusted_positions"][1]

    with pytest.raises(ValueError, match="Complete Digital Twin"):
        build_dt_only_agent_inputs(**data)
