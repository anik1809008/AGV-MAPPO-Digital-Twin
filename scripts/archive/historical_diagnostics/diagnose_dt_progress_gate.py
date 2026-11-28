"""Measure AoI gating and progress-monitor activation."""

import csv
from collections import Counter
from unittest.mock import patch

import src.marl.real_training_step as training
from scripts.run_four_method_evaluation import run_episode
from src.evaluation.model_loader import load_validation_models


def main():
    with open("results/four_method_eval_matrix.csv", newline="") as f:
        matrix = list(csv.DictReader(f))

    stats = Counter()
    original = training.update_dt_progress

    def observed(*args, **kwargs):
        aoi = kwargs["aoi"]

        stats["checks"] += 1
        stats["aoi_zero"] += aoi == 0
        stats["aoi_nonzero"] += aoi != 0
        stats["aoi_sum"] += aoi
        stats["aoi_max"] = max(stats["aoi_max"], aoi)

        result = original(*args, **kwargs)

        stats["stagnating"] += bool(result)
        if aoi == 0:
            stats["stagnating_at_fresh"] += bool(result)
        else:
            stats["stagnating_at_stale"] += bool(result)

        return result

    for latency in (0, 4):
        spec = next(
            r for r in matrix
            if int(r["latency_steps"]) == latency
            and float(r["immediate_probability"]) == 0.8
        )

        stats.clear()

        models = load_validation_models(
            method="M7",
            agent_count=int(spec["agent_count"]),
            seed=int(spec["seed"]),
            episodes_per_scenario=20,
        )

        with patch.object(
            training, "update_dt_progress", side_effect=observed
        ):
            outcome = run_episode(
                method="DT-PROGRESS-M7",
                specification=spec,
                models=models,
                max_steps=400,
            )

        checks = max(stats["checks"], 1)

        print(f"\nLATENCY {latency}")
        print("Episode:", spec["episode_id"])
        print("Success:", outcome["success"])
        print("Progress checks:", stats["checks"])
        print("Fresh telemetry (%):", round(
            100 * stats["aoi_zero"] / checks, 2
        ))
        print("Stale telemetry (%):", round(
            100 * stats["aoi_nonzero"] / checks, 2
        ))
        print("Mean AoI:", round(stats["aoi_sum"] / checks, 2))
        print("Maximum AoI:", stats["aoi_max"])
        print("Stagnation detected:", stats["stagnating"])
        print("Stagnation with fresh telemetry:",
              stats["stagnating_at_fresh"])
        print("Stagnation with stale telemetry:",
              stats["stagnating_at_stale"])


if __name__ == "__main__":
    main()
