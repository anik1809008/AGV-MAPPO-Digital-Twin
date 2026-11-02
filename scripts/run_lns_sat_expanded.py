"""Repeatable nominal eight-AGV LNS-SAT evaluation on MovingAI validation scenarios.

Uses different, nonoverlapping blocks of 8 scenario rows for each scenario ID.
All rows in the output identify both scenario_id and start_index.
"""
import argparse
import csv
import random
from pathlib import Path

from src.environment.simulator import GroundTruthSimulator
from src.evaluation.results_writer import RESULT_FIELDS
from src.mapf.lns_sat_runner import run_lns_sat_episode
from src.marl.scenario_split import VALIDATION_SCENARIO_IDS, get_random_scenario_path
from src.marl.training_instance import build_training_instance

MAP_PATH = "benchmarks/movingai/warehouse-10-20-10-2-1.map"
EXTRA_FIELDS = ["start_index", "planning_failed"]
CSV_FIELDS = list(RESULT_FIELDS) + EXTRA_FIELDS


def make_trials(count=50, agent_count=8, scenario_ids=VALIDATION_SCENARIO_IDS):
    """Construct reproducible, distinct (scenario, nonoverlapping row block) trials."""
    if count < 1 or agent_count < 1 or not scenario_ids:
        raise ValueError("count, agent_count, and scenario_ids must be positive/nonempty")
    return [
        (scenario_ids[i % len(scenario_ids)], (i // len(scenario_ids)) * agent_count)
        for i in range(count)
    ]


def evaluate_trial(scenario_id, start_index, max_iterations=100):
    instance = build_training_instance(
        map_path=MAP_PATH,
        scenario_path=get_random_scenario_path(scenario_id),
        agent_count=8,
        start_index=start_index,
    )
    simulator = GroundTruthSimulator(
        grid=instance["grid"],
        agent_positions=instance["starts"],
        agent_goals=instance["goals"],
    )
    result = run_lns_sat_episode(simulator=simulator, max_iterations=max_iterations)
    metrics = dict(result.get("metrics", {}))
    # The existing LNS-SAT runner has no metrics object on planning failures.
    metrics.setdefault("method", "LNS-SAT")
    metrics.setdefault("success", bool(result.get("all_goals_reached", False)))
    metrics.setdefault("collision", bool(result.get("collision", False)))
    metrics.setdefault("deadlock", bool(simulator.is_deadlocked()))
    metrics.setdefault("makespan", int(result.get("steps", 0)))
    metrics.setdefault("path_length", 0)
    metrics.setdefault("planning_time", "")
    metrics.setdefault("shield_interventions", 0)
    metrics.update(
        scenario_id=scenario_id,
        agent_count=8,
        start_index=start_index,
        seed="",  # No random-seed API is exposed by the existing LNS-SAT runner.
        uncertainty_condition="nominal",
        latency_steps=0,
        immediate_probability=1.0,
        planning_failed=bool(result.get("planning_failed", False)),
    )
    fingerprint = tuple((k, instance["starts"][k], instance["goals"][k]) for k in sorted(instance["starts"]))
    return metrics, fingerprint


def run_benchmark(count, output_path, max_iterations=100):
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"Refusing to append to/overwrite existing CSV: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    trials = make_trials(count=count)
    seen_instances = set()
    rows = []
    with output.open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for trial_index, (scenario_id, start_index) in enumerate(trials, 1):
            metrics, fingerprint = evaluate_trial(scenario_id, start_index, max_iterations)
            if fingerprint in seen_instances:
                raise ValueError(f"Duplicate AGV starts/goals at scenario={scenario_id}, start_index={start_index}")
            seen_instances.add(fingerprint)
            writer.writerow({field: metrics.get(field, "") for field in CSV_FIELDS})
            file.flush()
            rows.append(metrics)
            print(f"[{trial_index}/{count}] scenario={scenario_id} start={start_index} "
                  f"success={metrics['success']} collision={metrics['collision']} "
                  f"makespan={metrics['makespan']}", flush=True)
    successes = sum(bool(row["success"]) for row in rows)
    collisions = sum(bool(row["collision"]) for row in rows)
    failures = sum(bool(row["planning_failed"]) for row in rows)
    print(f"RESULTS: trials={len(rows)}, success={successes}, collision={collisions}, planning_failed={failures}")
    print(f"CSV: {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--results-path", default="results/lns_sat_expanded_50.csv")
    args = parser.parse_args()
    if args.trials < 1 or args.max_iterations < 1:
        parser.error("--trials and --max-iterations must be at least 1")
    run_benchmark(args.trials, args.results_path, args.max_iterations)


if __name__ == "__main__":
    main()
