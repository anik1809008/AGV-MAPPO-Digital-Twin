"""Tests for experimental DICR supervisor transitions."""

import pytest

from src.hybrid.dt_inferred_completion_supervisor import (
    DTInferredCompletionSupervisor,
)


def supervisor_in_m7():
    supervisor = DTInferredCompletionSupervisor()
    supervisor.fallback_to_m7(timestep=0)
    return supervisor


def activated_supervisor():
    supervisor = supervisor_in_m7()

    supervisor.confirm_inferred_completion(
        timestep=3,
        eligibility_confirmed=True,
        route_valid=True,
        aoi=2,
    )

    return supervisor


def observe(supervisor, timestep=4, **kwargs):
    return supervisor.observe(
        timestep=timestep,
        aoi_values={0: 2, 1: 2},
        possible_current_positions={
            0: {(0, 0)},
            1: {(3, 0)},
        },
        **kwargs,
    )


def test_original_uncertainty_switch_preserved():
    supervisor = DTInferredCompletionSupervisor()

    event = observe(supervisor)

    assert event.new_mode == "M7"
    assert event.reason == "aoi_threshold_exceeded"
    assert not supervisor.inferred_completion_active


def test_original_fresh_return_preserved():
    supervisor = supervisor_in_m7()

    for timestep in (1, 2, 3):
        decision = supervisor.observe(
            timestep=timestep,
            aoi_values={0: 0},
            possible_current_positions={0: {(0, 0)}},
        )

    assert decision == "RETURN_READY"

    supervisor.confirm_return(timestep=3)

    assert supervisor.mode == "LNS-SAT"
    assert not supervisor.inferred_completion_active


def test_inferred_completion_activation():
    supervisor = activated_supervisor()

    assert supervisor.mode == "LNS-SAT"
    assert supervisor.inferred_completion_active
    assert supervisor.events[-1].reason == (
        "dt_inferred_completion"
    )
    assert supervisor.last_switch_timestep == 3


def test_valid_route_keeps_completion_active():
    supervisor = activated_supervisor()

    event = observe(
        supervisor,
        inferred_route_valid=True,
    )

    assert event is None
    assert supervisor.mode == "LNS-SAT"
    assert supervisor.inferred_completion_active


@pytest.mark.parametrize("validation", [False, None])
def test_invalid_or_missing_validation_falls_back(validation):
    supervisor = activated_supervisor()

    event = observe(
        supervisor,
        inferred_route_valid=validation,
    )

    assert event.new_mode == "M7"
    assert event.reason == "dt_inferred_route_invalid"
    assert not supervisor.inferred_completion_active
    assert supervisor.reliable_steps == 0


def test_pending_commands_force_fallback():
    supervisor = activated_supervisor()

    event = observe(
        supervisor,
        inferred_route_valid=True,
        pending_commands=True,
    )

    assert event.new_mode == "M7"
    assert not supervisor.inferred_completion_active


def test_explicit_fallback_clears_completion():
    supervisor = activated_supervisor()

    supervisor.fallback_to_m7(timestep=4)

    assert supervisor.mode == "M7"
    assert not supervisor.inferred_completion_active


def test_reject_unconfirmed_eligibility():
    supervisor = supervisor_in_m7()

    with pytest.raises(ValueError):
        supervisor.confirm_inferred_completion(
            timestep=3,
            eligibility_confirmed=False,
            route_valid=True,
            aoi=2,
        )

    assert supervisor.mode == "M7"


def test_reject_unvalidated_route():
    supervisor = supervisor_in_m7()

    with pytest.raises(ValueError):
        supervisor.confirm_inferred_completion(
            timestep=3,
            eligibility_confirmed=True,
            route_valid=False,
            aoi=2,
        )


def test_minimum_dwell_required():
    supervisor = supervisor_in_m7()

    with pytest.raises(ValueError):
        supervisor.confirm_inferred_completion(
            timestep=1,
            eligibility_confirmed=True,
            route_valid=True,
            aoi=2,
        )


def test_activation_requires_m7():
    supervisor = DTInferredCompletionSupervisor()

    with pytest.raises(ValueError):
        supervisor.confirm_inferred_completion(
            timestep=3,
            eligibility_confirmed=True,
            route_valid=True,
            aoi=2,
        )


@pytest.mark.parametrize("aoi", [-1, 0, 3])
def test_reject_invalid_activation_aoi(aoi):
    supervisor = supervisor_in_m7()

    with pytest.raises(ValueError):
        supervisor.confirm_inferred_completion(
            timestep=3,
            eligibility_confirmed=True,
            route_valid=True,
            aoi=aoi,
        )
