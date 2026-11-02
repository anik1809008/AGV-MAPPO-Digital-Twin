import pytest

from src.hybrid.two_way_supervisor import TwoWayHybridSupervisor


def reliable(supervisor, timestep, pending=False):
    return supervisor.observe(
        timestep=timestep,
        aoi_values={0: 0, 1: 0},
        possible_current_positions={
            0: {(0, 0)},
            1: {(2, 0)},
        },
        pending_commands=pending,
    )


def test_normal_to_m7_on_uncertainty():
    supervisor = TwoWayHybridSupervisor()

    event = supervisor.observe(
        timestep=2,
        aoi_values={0: 2, 1: 2},
    )

    assert event.new_mode == "M7"
    assert supervisor.mode == "M7"


def test_return_requires_sustained_reliability():
    supervisor = TwoWayHybridSupervisor(
        recovery_steps=3,
        min_dwell_steps=3,
    )

    supervisor.observe(timestep=2, aoi_values={0: 2, 1: 2})

    assert reliable(supervisor, 3) is None
    assert reliable(supervisor, 4) is None
    assert reliable(supervisor, 5) == "RETURN_READY"

    # The mode does not change before replanning confirmation.
    assert supervisor.mode == "M7"

    event = supervisor.confirm_return(5)
    assert event.new_mode == "LNS-SAT"
    assert supervisor.mode == "LNS-SAT"


def test_pending_commands_block_return():
    supervisor = TwoWayHybridSupervisor(recovery_steps=2)
    supervisor.observe(timestep=2, aoi_values={0: 2})

    assert reliable(supervisor, 3, pending=True) is None
    assert reliable(supervisor, 4) is None
    assert reliable(supervisor, 5) == "RETURN_READY"


def test_unreliable_state_resets_recovery():
    supervisor = TwoWayHybridSupervisor(recovery_steps=2)
    supervisor.observe(timestep=2, aoi_values={0: 2})

    assert reliable(supervisor, 3) is None

    supervisor.observe(
        timestep=4,
        aoi_values={0: 0, 1: 0},
        possible_current_positions={
            0: {(0, 0), (1, 0)},
            1: {(2, 0)},
        },
    )

    assert reliable(supervisor, 5) is None
    assert reliable(supervisor, 6) == "RETURN_READY"


def test_initial_planning_failure_falls_back_to_m7():
    supervisor = TwoWayHybridSupervisor()
    event = supervisor.fallback_to_m7(timestep=0)

    assert supervisor.mode == "M7"
    assert event.reason == "lns_planning_failed"


def test_rejects_unconfirmed_return():
    supervisor = TwoWayHybridSupervisor()
    supervisor.fallback_to_m7(timestep=0)

    with pytest.raises(ValueError):
        supervisor.confirm_return(timestep=1)
