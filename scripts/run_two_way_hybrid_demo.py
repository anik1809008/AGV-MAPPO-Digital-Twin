"""Eight-AGV LNS-SAT -> M7 hybrid demonstration."""

import argparse
import random

import numpy as np
import torch

from src.communication.scheduled_latency_channel import ScheduledLatencyChannel
from src.environment.execution_delay import ExecutionDelayModel
from src.evaluation.context_factory import build_experiment_context
from src.evaluation.controller_factory import build_method_controllers
from src.evaluation.model_loader import load_validation_models
from src.hybrid.two_way_episode_runner import run_two_way_episode
from src.hybrid.two_way_supervisor import TwoWayHybridSupervisor
from src.marl.scenario_split import get_random_scenario_path
from src.marl.training_instance import build_training_instance


MAP_PATH = (
    "benchmarks/movingai/"
    "warehouse-10-20-10-2-1.map"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--scenario-id", type=int, default=19)
    parser.add_argument("--episodes-per-scenario", type=int, default=40)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--latency", type=int, default=2)
    parser.add_argument("--aoi-threshold", type=int, default=2)
    parser.add_argument("--immediate-probability", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=400)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    instance = build_training_instance(
        map_path=MAP_PATH,
        scenario_path=get_random_scenario_path(args.scenario_id),
        agent_count=8,
        start_index=args.start_index,
    )

    models = load_validation_models(
        method="M7",
        agent_count=8,
        seed=args.seed,
        episodes_per_scenario=args.episodes_per_scenario,
    )

    controllers = build_method_controllers(
        method="M7",
        actor=models["actor"],
        grid=instance["grid"],
    )

    context = build_experiment_context(
        grid=instance["grid"],
        starts=instance["starts"],
        goals=instance["goals"],
        actor=models["actor"],
        critic=models["critic"],
        latency_steps=args.latency,
        max_steps=args.max_steps,
        m7_controller=controllers["m7_controller"],
        m7_progress_monitor=controllers["m7_progress_monitor"],
        m7_recovery_planner=controllers["m7_recovery_planner"],
        execution_delay_model=ExecutionDelayModel(
            immediate_probability=args.immediate_probability,
            seed=args.seed,
        ),
    )

    context['telemetry_channel'] = ScheduledLatencyChannel(
        schedule=[(0, 0), (10, 2), (30, 0)]
    )

    supervisor = TwoWayHybridSupervisor(
        aoi_threshold=args.aoi_threshold,
    )

    result = run_two_way_episode(
        context=context,
        supervisor=supervisor,
    )

    print("\n===== TWO-WAY HYBRID EIGHT-AGV DEMONSTRATION =====")
    print("Checkpoint:", models["checkpoint_path"])
    print("Scenario:", args.scenario_id)
    print("Seed:", args.seed)
    print(
        "Telemetry schedule:",
        context["telemetry_channel"].schedule,
    )
    print("Execution immediate probability:",
          args.immediate_probability)

    for event in result["switch_events"]:
        print(
            "SWITCH:",
            f"t={event.timestep}",
            f"{event.previous_mode} -> {event.new_mode}",
            f"reason={event.reason}",
            f"agent={event.agent_id}",
            f"aoi={event.aoi}",
        )

    for key, value in result.items():
        if key != "switch_events":
            print(f"{key}: {value}")

    print("Final controller:", supervisor.mode)
    print(
        "Switch demonstrated:",
        bool(result["switch_events"])
        and result["lns_steps"] > 0
        and result["m7_steps"] > 0,
    )


if __name__ == "__main__":
    main()
