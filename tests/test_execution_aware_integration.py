from unittest.mock import patch

from src.environment.execution_delay import ExecutionDelayModel
from src.evaluation.context_factory import build_experiment_context
from src.hybrid.execution_aware_supervisor import (
    ExecutionAwareTwoWaySupervisor,
)
from src.hybrid.two_way_episode_runner import run_two_way_episode


def test_pending_lns_command_drains_before_m7_execution():
    context = build_experiment_context(
        grid=[[0, 0, 0, 0]],
        starts={0: (0, 0)},
        goals={0: (3, 0)},
        actor=None,
        critic=None,
        latency_steps=0,
        max_steps=10,
        m7_controller=object(),
        execution_delay_model=ExecutionDelayModel(
            immediate_probability=0.0,
            seed=1,
        ),
    )

    calls = []

    def fake_plan(**kwargs):
        return {
            "paths": {
                0: [(0, 0), (1, 0), (2, 0), (3, 0)]
            },
            "makespan": 3,
        }

    def fake_m7_step(**kwargs):
        position = kwargs["simulator"].agent_positions[0]
        calls.append((kwargs["current_timestep"], position))
        result = kwargs["simulator"].step_joint({0: 3})
        return {
            "actions": [3],
            "collision": result["collision"],
        }

    supervisor = ExecutionAwareTwoWaySupervisor(
        aoi_threshold=99,
        recovery_steps=99,
        min_dwell_steps=99,
    )

    with patch(
        "src.hybrid.two_way_episode_runner.solve_lns_sat",
        side_effect=fake_plan,
    ), patch(
        "src.hybrid.two_way_episode_runner.run_real_training_step",
        side_effect=fake_m7_step,
    ):
        result = run_two_way_episode(
            context,
            supervisor,
            enable_execution_handover=True,
        )

    assert result["all_goals_reached"]
    assert not result["collision"]
    assert result["lns_steps"] == 1
    assert calls
    # At timestep 1, the delayed LNS command is drained.
    # M7 first acts at timestep 2, starting from the drained position.
    assert calls[0] == (2, (1, 0))
    assert any(
        event.reason == "pending_execution_uncertainty"
        for event in result["switch_events"]
    )
