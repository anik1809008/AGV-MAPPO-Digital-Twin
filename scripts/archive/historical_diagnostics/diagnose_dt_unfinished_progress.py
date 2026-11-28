"""Diagnose DT progress tracking for unfinished AGVs."""

import csv
from collections import Counter
from unittest.mock import patch

import src.marl.real_training_step as training
from scripts.run_four_method_evaluation import run_episode
from src.evaluation.model_loader import load_validation_models


def main():
    with open("results/four_method_eval_matrix.csv", newline="") as f:
        matrix = list(csv.DictReader(f))

    spec = next(
        r for r in matrix
        if int(r["episode_id"]) == 10
        and int(r["latency_steps"]) == 4
        and float(r["immediate_probability"]) == 0.8
    )

    stats = {0: Counter(), 1: Counter()}
    original = training.update_dt_estimated_progress

    def observed(*args, **kwargs):
        agent_id = kwargs["agent_id"]

        if agent_id not in stats:
            return original(*args, **kwargs)

        monitor = kwargs["monitor"]
        candidates = set(kwargs["possible_positions"])
        goal = kwargs["goal"]
        data = stats[agent_id]

        data["checks"] += 1
        data["singleton"] += len(candidates) == 1
        data["ambiguous"] += len(candidates) > 1
        data["max_set_size"] = max(
            data["max_set_size"], len(candidates)
        )

        previous_best = monitor.best_distances.get(agent_id)

        result = original(*args, **kwargs)

        current_best = monitor.best_distances.get(agent_id)
        count = monitor.no_progress_counts.get(agent_id, 0)

        data["updates"] += len(candidates) == 1
        data["max_stagnation_count"] = max(
            data["max_stagnation_count"], count
        )
        data["stagnation_detected"] += bool(result)

        if (
            current_best is not None
            and (previous_best is None or current_best < previous_best)
        ):
            data["new_best_distance"] += 1

        data["final_best_distance"] = current_best
        data["final_stagnation_count"] = count

        return result

    models = load_validation_models(
        method="M7",
        agent_count=int(spec["agent_count"]),
        seed=int(spec["seed"]),
        episodes_per_scenario=20,
    )

    with patch.object(
        training,
        "update_dt_estimated_progress",
        side_effect=observed,
    ):
        outcome = run_episode(
            method="DT-ESTIMATED-PROGRESS-M7",
            specification=spec,
            models=models,
            max_steps=400,
        )

    print("\nEPISODE 10 — UNFINISHED AGV PROGRESS")
    print("Success:", outcome["success"])
    print("Steps:", outcome["makespan"])

    for agent_id, data in stats.items():
        print(f"\nAGV {agent_id}")
        for key in (
            "checks",
            "singleton",
            "ambiguous",
            "max_set_size",
            "updates",
            "new_best_distance",
            "max_stagnation_count",
            "final_stagnation_count",
            "final_best_distance",
            "stagnation_detected",
        ):
            print(f"{key}: {data[key]}")


if __name__ == "__main__":
    main()
