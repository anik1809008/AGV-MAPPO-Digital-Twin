"""Matched DT-Robust M7 evaluation at 400, 500, 700 steps."""

import argparse
import csv
import random
import time
from collections import defaultdict
from pathlib import Path

from scripts.run_four_method_evaluation import run_episode
from src.evaluation.model_loader import load_validation_models

MATRIX = Path("results/dt_hybrid_heldout1000_matrix.csv")
OUTDIR = Path("results/m7_horizon_comparison")
SELECTED = OUTDIR / "selected_100_episodes.csv"
DETAILS = OUTDIR / "episode_results.csv"
SUMMARY = OUTDIR / "summary.csv"

HORIZONS = (400, 500, 700)
METHOD = "DT-ROBUST-PROGRESS-M7"
SELECTION_SEED = 20261116

FIELDS = [
    "episode_id", "scenario_id", "start_index", "seed",
    "latency_steps", "immediate_probability",
    "max_steps", "success", "collision", "deadlock",
    "timeout", "makespan", "path_length",
    "planning_time", "wall_time", "cpu_time",
]


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def choose_episodes(rows):
    if SELECTED.exists():
        selected = read_csv(SELECTED)
        if len(selected) != 100:
            raise ValueError("Saved selection must contain 100 episodes")
        return selected

    rng = random.Random(SELECTION_SEED)
    groups = defaultdict(list)

    for row in rows:
        groups[row["scenario_id"]].append(row)

    # Equal representation of scenarios where possible.
    if len(groups) == 5 and all(len(g) >= 20 for g in groups.values()):
        selected = []
        for scenario in sorted(groups):
            selected.extend(rng.sample(groups[scenario], 20))
    else:
        selected = rng.sample(rows, 100)

    selected.sort(key=lambda r: str(r["episode_id"]))
    write_csv(SELECTED, selected, list(rows[0].keys()))
    return selected


def is_true(value):
    return str(value).lower() in ("true", "1")


def make_summary(rows):
    results = []

    for horizon in HORIZONS:
        group = [
            row for row in rows
            if int(row["max_steps"]) == horizon
        ]
        if not group:
            continue

        n = len(group)
        successes = sum(is_true(r["success"]) for r in group)
        timeouts = sum(is_true(r["timeout"]) for r in group)

        def mean(field):
            values = [
                float(r[field]) for r in group
                if str(r.get(field, "")).strip()
            ]
            return sum(values) / len(values) if values else 0.0

        results.append({
            "max_steps": horizon,
            "episodes": n,
            "successes": successes,
            "success_rate_pct": round(100 * successes / n, 2),
            "timeouts": timeouts,
            "mean_executed_steps": round(mean("makespan"), 2),
            "mean_wall_seconds": round(mean("wall_time"), 4),
            "total_wall_seconds": round(
                sum(float(r["wall_time"]) for r in group), 4
            ),
            "mean_cpu_seconds": round(mean("cpu_time"), 4),
            "mean_planning_seconds": round(mean("planning_time"), 4),
        })

    write_csv(SUMMARY, results, list(results[0].keys()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    if not 1 <= args.limit <= 100:
        parser.error("--limit must be between 1 and 100")

    OUTDIR.mkdir(parents=True, exist_ok=True)

    matrix = read_csv(MATRIX)
    if len(matrix) < 100:
        raise ValueError("Evaluation matrix contains fewer than 100 rows")

    selected = choose_episodes(matrix)[:args.limit]

    existing = read_csv(DETAILS) if DETAILS.exists() else []
    completed = {
        (r["episode_id"], int(r["max_steps"]))
        for r in existing
    }
    if len(completed) != len(existing):
        raise ValueError("Duplicate records in results file")

    models_cache = {}

    for specification in selected:
        seed = int(specification["seed"])
        agents = int(specification["agent_count"])

        key = (agents, seed)
        if key not in models_cache:
            models_cache[key] = load_validation_models(
                method="M7",
                agent_count=agents,
                seed=seed,
                episodes_per_scenario=20,
            )

        for horizon in HORIZONS:
            episode_id = specification["episode_id"]
            if (episode_id, horizon) in completed:
                continue

            wall_start = time.perf_counter()
            cpu_start = time.process_time()

            metrics = run_episode(
                METHOD,
                specification,
                models_cache[key],
                max_steps=horizon,
            )

            cpu_time = time.process_time() - cpu_start
            wall_time = time.perf_counter() - wall_start

            success = is_true(metrics["success"])
            collision = is_true(metrics.get("collision", False))
            deadlock = is_true(metrics.get("deadlock", False))
            makespan = int(metrics["makespan"])

            timeout = (
                not success
                and not collision
                and not deadlock
                and makespan >= horizon
            )

            row = {
                "episode_id": episode_id,
                "scenario_id": specification["scenario_id"],
                "start_index": specification["start_index"],
                "seed": seed,
                "latency_steps": specification["latency_steps"],
                "immediate_probability":
                    specification["immediate_probability"],
                "max_steps": horizon,
                "success": success,
                "collision": collision,
                "deadlock": deadlock,
                "timeout": timeout,
                "makespan": makespan,
                "path_length": metrics.get("path_length", ""),
                "planning_time": metrics.get("planning_time", ""),
                "wall_time": round(wall_time, 5),
                "cpu_time": round(cpu_time, 5),
            }

            # Save every successful evaluation immediately.
            with DETAILS.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                if f.tell() == 0:
                    writer.writeheader()
                writer.writerow(row)

            existing.append(row)
            completed.add((episode_id, horizon))
            make_summary(existing)

            print(
                f"Episode {episode_id} | "
                f"max={horizon} | "
                f"success={success} | "
                f"steps={makespan} | "
                f"wall={wall_time:.2f}s",
                flush=True,
            )

    print("\nSUMMARY")
    for row in read_csv(SUMMARY):
        print(row)

    print(f"\nResults: {DETAILS}")
    print(f"Summary: {SUMMARY}")


if __name__ == "__main__":
    main()
