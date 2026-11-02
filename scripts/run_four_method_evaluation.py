"""Matched four-method evaluation using the frozen M7 checkpoint."""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch

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

METHODS = (
    "LNS-SAT",
    "M7",
    "TWO-WAY",
    "RECOVERY-HYBRID",
)

FIELDS = (
    "method",
    "episode_id",
    "scenario_id",
    "start_index",
    "seed",
    "latency_steps",
    "immediate_probability",
    "success",
    "collision",
    "deadlock",
    "makespan",
    "path_length",
    "planning_time",
    "return_replans",
    "recovery_attempts",
    "recovery_pause_steps",
    "switch_count",
    "successful_returns",
    "planning_failures",
)


def run_episode(method, specification, models, max_steps=400):
    seed = int(specification["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    instance = build_training_instance(
        map_path=specification["map_path"],
        scenario_path=get_random_scenario_path(
            int(specification["scenario_id"])
        ),
        agent_count=int(specification["agent_count"]),
        start_index=int(specification["start_index"]),
    )

    probability = float(specification["immediate_probability"])

    if method == "LNS-SAT":
        metrics = run_lns_sat_uncertain_episode(
            grid=instance["grid"],
            starts=instance["starts"],
            goals=instance["goals"],
            immediate_probability=probability,
            seed=seed,
            max_steps=max_steps,
        )

        return {
            "success": metrics["success"],
            "collision": metrics["collision"],
            "deadlock": metrics["deadlock"],
            "makespan": metrics["makespan"],
            "path_length": metrics["path_length"],
            "planning_time": metrics["planning_time"],
        }

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
        latency_steps=int(specification["latency_steps"]),
        max_steps=max_steps,
        m7_controller=controllers["m7_controller"],
        m7_progress_monitor=controllers["m7_progress_monitor"],
        m7_recovery_planner=controllers["m7_recovery_planner"],
        execution_delay_model=ExecutionDelayModel(
            immediate_probability=probability,
            seed=seed,
        ),
    )

    if method == "M7":
        metrics = run_real_method(
            method="M7",
            latency_steps=int(specification["latency_steps"]),
            context=context,
        )

        return {
            "success": metrics["success"],
            "collision": metrics["collision"],
            "deadlock": metrics["deadlock"],
            "makespan": metrics["makespan"],
            "path_length": metrics["path_length"],
            "planning_time": metrics["planning_time"],
        }

    supervisor = TwoWayHybridSupervisor(aoi_threshold=2)

    metrics = run_two_way_episode(
        context=context,
        supervisor=supervisor,
        enable_recovery_pause=(method == "RECOVERY-HYBRID"),
    )

    return {
        "success": metrics["all_goals_reached"],
        "collision": metrics["collision"],
        "deadlock": metrics["deadlock"],
        "makespan": metrics["steps"],
        "path_length": metrics["path_length"],
        "planning_time": metrics["planning_time"],
        "return_replans": metrics["return_replans"],
        "recovery_attempts": metrics["recovery_attempts"],
        "recovery_pause_steps": metrics["recovery_pause_steps"],
        "switch_count": len(metrics["switch_events"]),
        "successful_returns": sum(
            event.previous_mode == "M7"
            and event.new_mode == "LNS-SAT"
            for event in metrics["switch_events"]
        ),
        "planning_failures": metrics["planning_failures"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix",
        default="results/four_method_eval_matrix.csv",
    )
    parser.add_argument(
        "--output",
        default="results/four_method_evaluation.csv",
    )
    parser.add_argument(
        "--method",
        choices=METHODS,
        required=True,
    )
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing evaluation output",
    )
    args = parser.parse_args()

    if args.limit < 1 or args.limit > 250:
        parser.error("--limit must be between 1 and 250")

    with open(args.matrix, newline="") as handle:
        specifications = list(csv.DictReader(handle))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    completed = set()
    expected = {
        str(row["episode_id"]): row
        for row in specifications
    }

    if len(expected) != len(specifications):
        raise ValueError("Duplicate episode_id in evaluation matrix")

    if output.exists():
        if not args.resume:
            raise FileExistsError(
                f"Output already exists: {output}; use --resume"
            )

        with output.open(newline="") as existing:
            reader = csv.DictReader(existing)
            if tuple(reader.fieldnames or ()) != FIELDS:
                raise ValueError("Existing output schema does not match")
            for row in reader:
                if row["method"] != args.method:
                    raise ValueError("Output contains a different method")
                episode_id = row["episode_id"]
                if episode_id in completed:
                    raise ValueError("Duplicate episode in output")

                if episode_id not in expected:
                    raise ValueError("Unknown episode in output")

                original = expected[episode_id]

                for field in (
                    "scenario_id",
                    "seed",
                    "start_index",
                    "latency_steps",
                    "immediate_probability",
                ):
                    if str(row[field]) != str(original[field]):
                        raise ValueError(
                            f"Episode {episode_id}: mismatch in {field}"
                        )

                completed.add(episode_id)

    with output.open("a" if output.exists() else "x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if not completed and handle.tell() == 0:
            writer.writeheader()

        for specification in specifications[:args.limit]:
            if str(specification["episode_id"]) in completed:
                continue
            models = None

            if args.method != "LNS-SAT":
                models = load_validation_models(
                    method="M7",
                    agent_count=int(specification["agent_count"]),
                    seed=int(specification["seed"]),
                    episodes_per_scenario=20,
                )

            metrics = run_episode(
                args.method,
                specification,
                models,
                max_steps=args.max_steps,
            )

            row = {
                "method": args.method,
                **{
                    key: specification[key]
                    for key in (
                        "episode_id",
                        "scenario_id",
                        "start_index",
                        "seed",
                        "latency_steps",
                        "immediate_probability",
                    )
                },
                **metrics,
            }

            writer.writerow({
                field: row.get(field, "") for field in FIELDS
            })
            handle.flush()

            print(
                f"{args.method} episode={row['episode_id']} "
                f"success={row['success']} "
                f"steps={row['makespan']}",
                flush=True,
            )

    print("Saved:", output)


if __name__ == "__main__":
    main()
