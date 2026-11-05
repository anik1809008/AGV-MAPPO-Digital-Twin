from src.hybrid.execution_aware_supervisor import (
    ExecutionAwareTwoWaySupervisor,
)


def observe(supervisor, pending=False):
    return supervisor.observe(
        timestep=1,
        aoi_values={0: 0, 1: 0},
        planned_next_positions={0: (0, 1)},
        possible_current_positions={
            0: {(0, 0)},
            1: {(2, 0)},
        },
        pending_commands=pending,
    )


def test_no_execution_risk_keeps_classical_mode():
    supervisor = ExecutionAwareTwoWaySupervisor()

    event = observe(supervisor, pending=False)

    assert event is None
    assert supervisor.mode == "LNS-SAT"


def test_pending_commands_trigger_m7():
    supervisor = ExecutionAwareTwoWaySupervisor()

    event = observe(supervisor, pending=True)

    assert event is not None
    assert event.reason == "pending_execution_uncertainty"
    assert event.previous_mode == "LNS-SAT"
    assert event.new_mode == "M7"
    assert supervisor.mode == "M7"
    assert supervisor.last_switch_timestep == 1


def test_switch_is_recorded():
    supervisor = ExecutionAwareTwoWaySupervisor()

    observe(supervisor, pending=True)

    assert len(supervisor.events) == 1
    assert supervisor.events[0].reason == (
        "pending_execution_uncertainty"
    )
