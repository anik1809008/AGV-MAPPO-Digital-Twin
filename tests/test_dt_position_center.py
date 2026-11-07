import pytest

from src.marl.dt_position_center import (
    select_dt_observation_centers,
)


def test_singleton_estimate_used_instead_of_stale_telemetry():
    centers = select_dt_observation_centers(
        trusted_positions={0: (1, 0)},
        possible_current_positions={0: {(4, 0)}},
    )

    assert centers[0] == (4, 0)


def test_uncertain_estimate_uses_trusted_position():
    centers = select_dt_observation_centers(
        trusted_positions={0: (1, 0)},
        possible_current_positions={
            0: {(1, 0), (2, 0), (3, 0)}
        },
    )

    assert centers[0] == (1, 0)


def test_missing_position_estimates_rejected():
    with pytest.raises(ValueError):
        select_dt_observation_centers(
            trusted_positions={0: (1, 0)},
            possible_current_positions={0: set()},
        )
