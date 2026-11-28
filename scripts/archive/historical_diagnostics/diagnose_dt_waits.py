"""Diagnostic-only analysis of DT-observed M7 action selection."""

import csv
from collections import Counter
from statistics import mean
from unittest.mock import patch

from src.evaluation.model_loader import load_validation_models
from src.safety.m7_controller import M7Controller
from scripts.run_four_method_evaluation import run_episode


def main():
    with open("results/four_method_eval_matrix.csv", newline="") as f:
        rows = list(csv.DictReader(f))

    original = M7Controller.select_action

    for latency in (0, 4):
        matching = [
            r for r in rows
            if int(r["latency_steps"]) == latency
            and float(r["immediate_probability"]) == 1.0
        ]

        if not matching:
            raise ValueError(f"No matching latency={latency}")

        spec = matching[0]
        counts = Counter()
        sizes = []

        def observed(self, *args, **kwargs):
            candidates = kwargs["possible_current_positions"]
            sizes.append(len(candidates))

            safety_args = {
                key: kwargs[key]
                for key in (
                    "agent_id",
                    "possible_current_positions",
                    "other_current_positions",
                    "other_next_positions",
                    "reachable_occupancies",
                    "other_possible_transitions",
                )
            }

            safe = sum(
                bool(self.shield.is_action_safe(
                    action=action,
                    **safety_args,
                ))
                for action in range(5)
            )

            counts["decisions"] += 1

            if safe == 0:
                counts["no_safe_action"] += 1

            action, probability = original(
                self, *args, **kwargs
            )

            if int(action) == 0:
                counts["wait"] += 1

            return action, probability

        models = load_validation_models(
            method="M7",
            agent_count=int(spec["agent_count"]),
            seed=int(spec["seed"]),
            episodes_per_scenario=20,
        )

        with patch.object(
            M7Controller, "select_action", observed
        ):
            result = run_episode(
                method="DT-OBSERVED-M7",
                specification=spec,
                models=models,
                max_steps=400,
            )

        print(f"\nLatency: {latency}")
        print("Episode:", spec["episode_id"])
        print("Success:", result["success"])
        print("Steps:", result["makespan"])
        print("Total decisions:", counts["decisions"])
        print("WAIT decisions:", counts["wait"])
        print(
            "WAIT percentage:",
            round(
                100 * counts["wait"] /
                max(counts["decisions"], 1), 2
            ),
        )
        print(
            "No safe action:",
            counts["no_safe_action"],
        )
        print(
            "Mean possible positions:",
            round(mean(sizes), 2) if sizes else 0,
        )
        print(
            "Maximum possible positions:",
            max(sizes, default=0),
        )


if __name__ == "__main__":
    main()
