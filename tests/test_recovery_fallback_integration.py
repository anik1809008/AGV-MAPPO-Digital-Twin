"""Integration test for experimental DT recovery cancellation."""

from unittest.mock import patch

from src.hybrid.two_way_episode_runner import run_two_way_episode
from src.hybrid.two_way_supervisor import TwoWayHybridSupervisor
from src.hybrid.execution_aware_handover import ExecutionAwareHandover
from src.environment.execution_delay import ExecutionDelayModel
from src.evaluation.context_factory import build_experiment_context


def test_rejected_recovery_does_not_queue_during_handover():
    context = build_experiment_context(
        grid=[[0] * 20],
        starts={0: (0, 0)},
        goals={0: (19, 0)},
        actor=None,
        critic=None,
        latency_steps=0,
        max_steps=25,
        m7_controller=object(),
        execution_delay_model=ExecutionDelayModel(
            immediate_probability=1.0,
            seed=1,
        ),
    )

    supervisor = TwoWayHybridSupervisor(
        aoi_threshold=2,
        recovery_steps=2,
        min_dwell_steps=2,
    )

    rejected = []
    handover_steps = []
    queued_steps = []

    executor = context["delayed_executor"]
    original_queue = executor.queue_commands
    original_handover_step = ExecutionAwareHandover.step

    def tracked_queue(actions, current_timestep):
        queued_steps.append(current_timestep)
        return original_queue(actions, current_timestep)

    def tracked_handover(self, simulator, executor, timestep):
        handover_steps.append(timestep)
        return original_handover_step(
            self, simulator, executor, timestep
        )

    def injected_guard(plan, dt, index, pending_commands):
        if plan.get("single_agv_completion") and index == 1:
            rejected.append(index)
            return False
        return original_guard(plan, dt, index, pending_commands)

    def fake_m7_step(**kwargs):
        result = kwargs["simulator"].step_joint({0: 3})
        return {
            "actions": [3],
            "collision": result["collision"],
        }

    import src.hybrid.two_way_episode_runner as runner
    original_guard = runner.recovery_route_is_valid

    with patch.object(
        runner, "solve_lns_sat", return_value=None
    ), patch.object(
        runner, "run_real_training_step",
        side_effect=fake_m7_step
    ), patch.object(
        runner, "recovery_route_is_valid",
        side_effect=injected_guard
    ), patch.object(
        ExecutionAwareHandover, "step",
        new=tracked_handover
    ), patch.object(
        executor, "queue_commands",
        side_effect=tracked_queue
    ):
        result = run_two_way_episode(
            context=context,
            supervisor=supervisor,
            enable_execution_handover=True,
            enable_single_agv_completion=True,
        )

    assert rejected, "Recovery rejection was not exercised"
    assert handover_steps, "Handover was not exercised"
    assert not set(queued_steps).intersection(handover_steps), (
        "New recovery commands were queued during handover"
    )

    assert any(
        event.previous_mode == "LNS-SAT"
        and event.new_mode == "M7"
        for event in result["switch_events"]
    )

    assert result["collision"] is False
    assert result["deadlock"] is False
