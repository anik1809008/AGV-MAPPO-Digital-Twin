"""Diagnostic: compare DT estimates, issued commands, and real movement."""

import csv
from collections import Counter
from statistics import mean
from unittest.mock import patch

from src.evaluation.model_loader import load_validation_models
from scripts.run_four_method_evaluation import run_episode
import src.marl.real_training_step as training_step


def main():
    with open("results/four_method_eval_matrix.csv", newline="") as f:
        matrix = list(csv.DictReader(f))

    original = training_step.run_real_training_step

    for latency in (4,):
        spec = next(
            r for r in matrix
            if int(r["latency_steps"]) == latency
            and float(r["immediate_probability"]) == 0.8
        )

        stats = Counter()
        mismatches = []
        first = {}
        last = {}
        goals = {}

        def observed(**kwargs):
            simulator = kwargs["simulator"]
            before = dict(simulator.agent_positions)
            trusted = kwargs["trusted_positions"]
            goals.update(simulator.agent_goals)

            if not first:
                first.update(before)

            for agent_id, position in before.items():
                mismatches.append(
                    position != trusted[agent_id]
                )

            result = original(**kwargs)
            after = dict(simulator.agent_positions)
            last.update(after)

            for agent_id, position in before.items():
                issued = int(result["actions"][agent_id])
                executed = int(result["executed_actions"][agent_id])

                stats["commands"] += 1
                stats["issued_moves"] += issued != 0
                stats["executed_moves"] += executed != 0
                stats["actual_moves"] += after[agent_id] != position

                if issued != 0 and after[agent_id] == position:
                    stats["move_without_progress"] += 1

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
            result = run_episode(
                method="DT-INFERRED-M7",
                specification=spec,
                models=models,
                max_steps=800,
            )

        print(f"\nLATENCY {latency}")
        print("Episode:", spec["episode_id"])
        print("Success:", result["success"])
        print("Steps:", result["makespan"])
        print("Position mismatch (%):", round(100 * mean(mismatches), 2))
        print("Issued movement commands:", stats["issued_moves"])
        print("Executed movement commands:", stats["executed_moves"])
        print("Actual AGV movements:", stats["actual_moves"])
        print("Issued movement without movement:", stats["move_without_progress"])
        print("Agents at goal:", sum(last[a] == goals[a] for a in goals))

        for agent_id in sorted(goals):
            print(
                f"AGV {agent_id}: "
                f"start={first[agent_id]}, "
                f"final={last[agent_id]}, "
                f"goal={goals[agent_id]}"
            )


if __name__ == "__main__":
    main()
