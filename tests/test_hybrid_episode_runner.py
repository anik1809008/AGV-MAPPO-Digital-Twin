"""Lightweight hybrid integration checks using the real simulator/DT infrastructure."""
from unittest.mock import patch

from src.evaluation.context_factory import build_experiment_context
from src.environment.execution_delay import ExecutionDelayModel
from src.hybrid.episode_runner import run_hybrid_episode
from src.hybrid.supervisor import HybridSupervisor


class PlaceholderController:
    pass


def context(latency, length=3):
    return build_experiment_context(
        grid=[[0] * length], starts={0: (0, 0)}, goals={0: (length - 1, 0)},
        actor=None, critic=None, latency_steps=latency,
        max_steps=5, m7_controller=PlaceholderController(),
        execution_delay_model=ExecutionDelayModel(
            immediate_probability=1.0, seed=1,
        ),
    )


def plan(**kwargs):
    return {"paths": {0: [(0, 0), (1, 0), (2, 0)]}, "makespan": 2}


def test_normal_path_reaches_goal_without_switch():
    c = context(latency=0)
    with patch("src.hybrid.episode_runner.solve_lns_sat", side_effect=plan):
        outcome = run_hybrid_episode(c, HybridSupervisor(aoi_threshold=2))
    assert outcome["all_goals_reached"]
    assert outcome["lns_steps"] == 2
    assert outcome["m7_steps"] == 0
    assert outcome["switch_events"] == []


def test_switch_calls_real_m7_step_interface_without_reset():
    c = context(latency=2, length=5)
    called = []

    def m7_step(**kwargs):
        called.append((kwargs["current_timestep"],
                       dict(kwargs["simulator"].agent_positions)))
        kwargs["simulator"].step_joint({0: 3})  # EAST in existing Action enum
        return {"actions": [3], "collision": False}

    with patch("src.hybrid.episode_runner.solve_lns_sat", return_value={"paths": {0: [(x, 0) for x in range(5)]}, "makespan": 4}), \
         patch("src.hybrid.episode_runner.run_real_training_step",
               side_effect=m7_step):
        outcome = run_hybrid_episode(c, HybridSupervisor(aoi_threshold=2))
    assert len(outcome["switch_events"]) == 1
    assert outcome["switch_events"][0].timestep == 2
    assert outcome["lns_steps"] == 2
    assert outcome["m7_steps"] >= 1
    assert called[0][0] == 2
    assert called[0][1] == {0: (2, 0)}
