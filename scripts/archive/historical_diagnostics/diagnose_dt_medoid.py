"""Diagnose medoid centering and AGV movement without changing controllers."""

import csv
from collections import Counter
from unittest.mock import patch

import src.marl.real_training_step as training_step
from scripts.run_four_method_evaluation import run_episode
from src.evaluation.model_loader import load_validation_models
from src.marl.dt_medoid_center import select_dt_medoid_centers


def main():
    with open("results/four_method_eval_matrix.csv", newline="") as f:
        matrix = list(csv.DictReader(f))

    spec = next(
        row for row in matrix
        if int(row["episode_id"]) == 10
        and int(row["latency_steps"]) == 4
        and float(row["immediate_probability"]) == 0.8
    )

    original = training_step.run_real_training_step

    for method in ("DT-INFERRED-M7", "DT-MEDOID-M7"):
        stats = Counter()

        def observed(**kwargs):
            simulator = kwargs["simulator"]
            before = dict(simulator.agent_positions)
            trusted = kwargs["trusted_positions"]
            possible = kwargs["possible_current_positions"]

            medoid = select_dt_medoid_centers(trusted, possible)

            for agent_id in trusted:
                stats["checks"] += 1
                stats["medoid_differs"] += (
                    medoid[agent_id] != tuple(trusted[agent_id])
                )
                stats["multi_position"] += len(possible[agent_id]) > 1
                stats["medoid_differs_multi"] += (
                    len(possible[agent_id]) > 1
                    and medoid[agent_id] != tuple(trusted[agent_id])
                )

            result = original(**kwargs)
            after = dict(simulator.agent_positions)

            for agent_id in before:
                stats["issued_moves"] += int(result["actions"][agent_id]) != 0
                stats["actual_moves"] += (
                    after[agent_id] != before[agent_id]
                )

            return result

        models = load_validation_models(
            method="M7",
            agent_count=int(spec["agent_count"]),
            seed=int(spec["seed"]),
            episodes_per_scenario=20,
        )

        with patch.object(
            training_step,
            "run_real_training_step",
            side_effect=observed,
        ):
            outcome = run_episode(
                method=method,
                specification=spec,
                models=models,
                max_steps=400,
            )

        checks = max(stats["checks"], 1)
        multi = max(stats["multi_position"], 1)

        print(f"\nMethod: {method}")
        print("Success:", outcome["success"])
        print("Steps:", outcome["makespan"])
        print("Collision:", outcome["collision"])
        print("Medoid differs from trusted (%):", round(
            100 * stats["medoid_differs"] / checks, 2
        ))
        print("Medoid differs in ambiguous states (%):", round(
            100 * stats["medoid_differs_multi"] / multi, 2
        ))
        print("Issued movement commands:", stats["issued_moves"])
        print("Actual AGV movements:", stats["actual_moves"])


if __name__ == "__main__":
    main()
