"""Matched three-method research pilot.

LNS-SAT: open-loop planner with delayed execution.
M7: Digital Twin + MAPPO + safety control.
TWO-WAY: adaptive LNS-SAT/M7 with scheduled telemetry.
"""

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch

from src.communication.scheduled_latency_channel import ScheduledLatencyChannel
from src.environment.execution_delay import ExecutionDelayModel
from src.evaluation.context_factory import build_experiment_context
from src.evaluation.controller_factory import build_method_controllers
from src.evaluation.model_loader import load_validation_models
from src.evaluation.real_method_callback import run_real_method
from src.hybrid.two_way_episode_runner import run_two_way_episode
from src.hybrid.two_way_supervisor import TwoWayHybridSupervisor
from src.mapf.lns_sat_uncertain_runner import run_lns_sat_uncertain_episode
from src.marl.scenario_split import get_random_scenario_path
from src.marl.training_instance import build_training_instance

MAP_PATH = "benchmarks/movingai/warehouse-10-20-10-2-1.map"
SCHEDULE = ((0, 0), (10, 2), (30, 0))
METHODS = ("LNS-SAT", "M7", "TWO-WAY")

FIELDS = (
    "method", "scenario_id", "start_index", "seed",
    "immediate_probability", "telemetry_schedule",
    "success", "collision", "deadlock", "planning_failed",
    "makespan", "path_length", "episode_wall_time",
    "lns_planning_time", "lns_steps", "m7_steps",
    "switch_count", "return_replans", "planning_failures",
)


def run_trial(method, instance, args, models):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    started = time.perf_counter()

    if method == "LNS-SAT":
        result = run_lns_sat_uncertain_episode(
            grid=instance["grid"],
            starts=instance["starts"],
            goals=instance["goals"],
            immediate_probability=args.immediate_probability,
            seed=args.seed,
            max_steps=args.max_steps,
        )
        extras = {
            "lns_planning_time": result["planning_time"],
            "planning_failures": int(result["planning_failed"]),
        }
    else:
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
            latency_steps=0,
            max_steps=args.max_steps,
            m7_controller=controllers["m7_controller"],
            m7_progress_monitor=controllers["m7_progress_monitor"],
            m7_recovery_planner=controllers["m7_recovery_planner"],
            execution_delay_model=ExecutionDelayModel(
                immediate_probability=args.immediate_probability,
                seed=args.seed,
            ),
        )
        context["telemetry_channel"] = ScheduledLatencyChannel(
            schedule=SCHEDULE,
        )

        if method == "M7":
            metrics = run_real_method(
                method="M7",
                latency_steps=0,
                context=context,
            )
            result = {
                "success": metrics["success"],
                "collision": metrics["collision"],
                "deadlock": metrics["deadlock"],
                "planning_failed": False,
                "makespan": metrics["makespan"],
                "path_length": metrics["path_length"],
            }
            extras = {}
        else:
            supervisor = TwoWayHybridSupervisor(
                aoi_threshold=args.aoi_threshold,
            )
            hybrid = run_two_way_episode(
                context=context,
                supervisor=supervisor,
            )
            result = {
                "success": hybrid["all_goals_reached"],
                "collision": hybrid["collision"],
                "deadlock": hybrid["deadlock"],
                "planning_failed": hybrid["planning_failed"],
                "makespan": hybrid["steps"],
                "path_length": hybrid["path_length"],
            }
            extras = {
                "lns_planning_time": hybrid["planning_time"],
                "lns_steps": hybrid["lns_steps"],
                "m7_steps": hybrid["m7_steps"],
                "switch_count": len(hybrid["switch_events"]),
                "return_replans": hybrid["return_replans"],
                "planning_failures": hybrid["planning_failures"],
            }

    wall_time = time.perf_counter() - started
    return {**result, **extras, "episode_wall_time": wall_time}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episodes-per-scenario", type=int, default=40)
    parser.add_argument("--immediate-probability", type=float, default=1.0)
    parser.add_argument("--aoi-threshold", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument(
        "--results-path",
        default="results/controlled_three_method_pilot.csv",
    )
    args = parser.parse_args()

    if not 1 <= args.trials <= 3:
        parser.error("Pilot trials must be between 1 and 3")
    if not 0 <= args.immediate_probability <= 1:
        parser.error("Immediate probability must be between 0 and 1")

    output = Path(args.results_path)
    if output.exists():
        parser.error("Results file already exists; choose another path")
    output.parent.mkdir(parents=True, exist_ok=True)

    models = load_validation_models(
        method="M7",
        agent_count=8,
        seed=args.seed,
        episodes_per_scenario=args.episodes_per_scenario,
    )

    with output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()

        for i in range(args.trials):
            scenario_id = 18 + i
            instance = build_training_instance(
                map_path=MAP_PATH,
                scenario_path=get_random_scenario_path(scenario_id),
                agent_count=8,
                start_index=0,
            )

            for method in METHODS:
                result = run_trial(method, instance, args, models)
                row = {
                    "method": method,
                    "scenario_id": scenario_id,
                    "start_index": 0,
                    "seed": args.seed,
                    "immediate_probability": args.immediate_probability,
                    "telemetry_schedule": (
                        "not_observed_open_loop"
                        if method == "LNS-SAT"
                        else str(SCHEDULE)
                    ),
                    **result,
                }
                writer.writerow({
                    field: row.get(field, "") for field in FIELDS
                })
                handle.flush()
                print(
                    f"[{i + 1}/{args.trials}] {method}: "
                    f"success={row['success']} "
                    f"collision={row['collision']} "
                    f"steps={row['makespan']} "
                    f"wall={row['episode_wall_time']:.3f}s",
                    flush=True,
                )

    print("Results:", output)
    print(
        "NOTE: LNS-SAT is open-loop; M7 and TWO-WAY consume "
        "scheduled telemetry. LNS planning time and total wall "
        "time are distinct measurements."
    )


if __name__ == "__main__":
    main()
