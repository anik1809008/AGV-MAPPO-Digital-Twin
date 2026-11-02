import pytest

from src.hybrid.recovery_trigger import should_start_recovery


def check(mode="M7", timestep=10, last_switch=5, aoi=None):
    return should_start_recovery(
        mode=mode,
        timestep=timestep,
        last_switch_timestep=last_switch,
        aoi_values={0: 0, 1: 0} if aoi is None else aoi,
        min_dwell_steps=3,
    )


def test_recovery_starts_after_reliable_telemetry():
    assert check()


def test_recovery_rejected_with_stale_telemetry():
    assert not check(aoi={0: 0, 1: 2})


def test_recovery_rejected_before_dwell_time():
    assert not check(timestep=6)


def test_recovery_rejected_outside_m7():
    assert not check(mode="LNS-SAT")


def test_recovery_rejected_without_telemetry():
    assert not check(aoi={})


def test_recovery_rejected_without_previous_switch():
    assert not check(last_switch=None)


def test_invalid_dwell_time():
    with pytest.raises(ValueError):
        should_start_recovery(
            mode="M7",
            timestep=10,
            last_switch_timestep=5,
            aoi_values={0: 0},
            min_dwell_steps=0,
        )
