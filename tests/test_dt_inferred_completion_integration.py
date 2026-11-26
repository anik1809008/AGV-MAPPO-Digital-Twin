"""Integration tests for experimental DICR execution."""

from unittest.mock import patch

import pytest

import src.hybrid.two_way_episode_runner as runner

from src.environment.actions import Action
from src.environment.execution_delay import ExecutionDelayModel
from src.evaluation.context_factory import build_experiment_context
from src.hybrid.execution_aware_handover import ExecutionAwareHandover
from src.hybrid.execution_aware_supervisor import ExecutionAwareTwoWaySupervisor
from src.hybrid.dt_inferred_completion_supervisor import (
    DTInferredCompletionSupervisor,
)


def make_context(
    latency=2,
    probability=1.0,
    unfinished=1,
    max_steps=25,
):
    return build_experiment_context(
        grid=[[0] * 8],
        starts={
            0: (0, 0),
            1: (7, 0),
        },
        goals={
            0: (3, 0),
            1: (5, 0) if unfinished == 2 else (7, 0),
        },
        actor=None,
        critic=None,
        latency_steps=latency,
        max_steps=max_steps,
        m7_controller=object(),
        execution_delay_model=ExecutionDelayModel(
            immediate_probability=probability,
            seed=1,
        ),
    )


def fake_m7_step(**kwargs):
    simulator = kwargs["simulator"]

    actions = {
        a: int(Action.WAIT)
        for a in simulator.agent_positions
    }

    result = simulator.step_joint(actions)

    return {
        "actions": [
            actions[a] for a in sorted(actions)
        ],
        "collision": result["collision"],
    }


def dicr_activated(result):
    return any(
        event.reason == "dt_inferred_completion"
        for event in result["switch_events"]
    )


def execute(context, enabled=True):
    supervisor = (
        DTInferredCompletionSupervisor()
        if enabled
        else ExecutionAwareTwoWaySupervisor()
    )

    with patch.object(
        runner, "solve_lns_sat", return_value=None
    ), patch.object(
        runner, "run_real_training_step",
        side_effect=fake_m7_step,
    ):
        return runner.run_two_way_episode(
            context=context,
            supervisor=supervisor,
            enable_execution_handover=True,
            enable_single_agv_completion=True,
            enable_two_agv_completion=True,
            enable_dt_inferred_completion=enabled,
        )


@pytest.mark.parametrize("latency", [1, 2])
@pytest.mark.parametrize("unfinished", [1, 2])
def test_dicr_completes_under_delayed_telemetry(
    latency, unfinished
):
    context = make_context(
        latency=latency,
        unfinished=unfinished,
    )

    result = execute(context)

    assert dicr_activated(result)
    assert result["all_goals_reached"]
    assert not result["collision"]
    assert not result["deadlock"]

    assert context["simulator"].agent_positions == (
        context["simulator"].agent_goals
    )

    events = [
        e for e in result["switch_events"]
        if e.reason == "dt_inferred_completion"
    ]

    assert len(events) == 1
    assert events[0].aoi == latency
    assert events[0].timestep >= 3


def test_disabled_dicr_preserves_original_behavior():
    context = make_context(latency=2)

    result = execute(context, enabled=False)

    assert not dicr_activated(result)
    assert not result["all_goals_reached"]
    assert result["return_replans"] == 0
    assert not result["collision"]


def test_stochastic_execution_blocks_dicr():
    context = make_context(
        latency=2,
        probability=0.8,
    )

    result = execute(context)

    assert not dicr_activated(result)
    assert not result["collision"]


def test_insufficient_horizon_blocks_dicr():
    context = make_context(
        latency=2,
        max_steps=5,
    )

    result = execute(context)

    assert not dicr_activated(result)
    assert not result["all_goals_reached"]
    assert not result["collision"]


def test_invalid_route_triggers_safe_handover():
    context = make_context(
        latency=2,
        max_steps=15,
    )

    supervisor = DTInferredCompletionSupervisor()

    rejected = []
    queued_steps = []
    handover_steps = []

    executor = context["delayed_executor"]

    original_guard = (
        runner.inferred_recovery_route_is_valid
    )

    original_queue = executor.queue_commands
    original_handover = ExecutionAwareHandover.step

    def injected_guard(**kwargs):
        if kwargs["index"] == 1:
            rejected.append(kwargs["index"])
            return False

        return original_guard(**kwargs)

    def tracked_queue(actions, current_timestep):
        queued_steps.append(current_timestep)

        return original_queue(
            actions,
            current_timestep,
        )

    def tracked_handover(
        self, simulator, executor, timestep
    ):
        handover_steps.append(timestep)

        return original_handover(
            self,
            simulator,
            executor,
            timestep,
        )

    with patch.object(
        runner, "solve_lns_sat", return_value=None
    ), patch.object(
        runner, "run_real_training_step",
        side_effect=fake_m7_step,
    ), patch.object(
        runner, "inferred_recovery_route_is_valid",
        side_effect=injected_guard,
    ), patch.object(
        executor, "queue_commands",
        side_effect=tracked_queue,
    ), patch.object(
        ExecutionAwareHandover, "step",
        new=tracked_handover,
    ):
        result = runner.run_two_way_episode(
            context=context,
            supervisor=supervisor,
            enable_execution_handover=True,
            enable_single_agv_completion=True,
            enable_two_agv_completion=True,
            enable_dt_inferred_completion=True,
        )

    assert dicr_activated(result)
    assert rejected
    assert handover_steps
    assert queued_steps

    assert not set(queued_steps).intersection(
        handover_steps
    )

    assert any(
        event.reason == "dt_inferred_route_invalid"
        and event.new_mode == "M7"
        for event in result["switch_events"]
    )

    assert not supervisor.inferred_completion_active
    assert not result["collision"]
