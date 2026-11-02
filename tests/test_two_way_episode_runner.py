"""Deterministic integration tests; M7 step is stubbed, not retrained."""
from unittest.mock import patch

from src.evaluation.context_factory import build_experiment_context
from src.environment.execution_delay import ExecutionDelayModel
from src.hybrid.two_way_episode_runner import run_two_way_episode
from src.hybrid.two_way_supervisor import TwoWayHybridSupervisor


class PlaceholderController:
    pass


def context(latency=2, length=9):
    return build_experiment_context(
        grid=[[0] * length], starts={0: (0, 0)},
        goals={0: (length - 1, 0)},
        actor=None, critic=None, latency_steps=latency,
        max_steps=25, m7_controller=PlaceholderController(),
        execution_delay_model=ExecutionDelayModel(
            immediate_probability=1.0, seed=1,
        ),
    )


def planner(**kwargs):
    x = kwargs["starts"][0][0]
    goal = kwargs["goals"][0][0]
    return {"paths": {0: [(i, 0) for i in range(x, goal + 1)]},
            "makespan": goal - x}


def test_initial_planning_failure_falls_back_to_real_m7_step_interface():
    c = context(latency=2, length=4)

    def m7_step(**kwargs):
        kwargs["simulator"].step_joint({0: 3})
        return {"actions": [3], "collision": False}

    with patch("src.hybrid.two_way_episode_runner.solve_lns_sat", return_value=None), \
         patch("src.hybrid.two_way_episode_runner.run_real_training_step", side_effect=m7_step):
        result = run_two_way_episode(c, TwoWayHybridSupervisor())

    assert result["all_goals_reached"]
    assert result["lns_steps"] == 0
    assert result["m7_steps"] > 0
    assert result["switch_events"][0].reason == "lns_planning_failed"
    assert result["planning_failures"] >= 1


def test_switch_back_replans_from_dt_observation_after_recovery():
    c = context(latency=2)
    planner_starts = []

    def captured_plan(**kwargs):
        planner_starts.append(dict(kwargs["starts"]))
        return planner(**kwargs)

    def m7_step(**kwargs):
        # Simulate restoration of reliable low-latency communication.
        c["telemetry_channel"].latency_steps = 0
        kwargs["simulator"].step_joint({0: 3})
        return {"actions": [3], "collision": False}

    sup = TwoWayHybridSupervisor(aoi_threshold=2, recovery_steps=2,
                                 min_dwell_steps=2)
    with patch("src.hybrid.two_way_episode_runner.solve_lns_sat", side_effect=captured_plan), \
         patch("src.hybrid.two_way_episode_runner.run_real_training_step", side_effect=m7_step):
        result = run_two_way_episode(c, sup)

    assert result["all_goals_reached"]
    assert result["lns_steps"] > 2
    assert result["m7_steps"] >= 1
    assert result["return_replans"] >= 1
    assert any(e.previous_mode == "M7" and e.new_mode == "LNS-SAT"
               for e in result["switch_events"])
    assert len(planner_starts) >= 2
    assert planner_starts[1] != planner_starts[0]
