"""Count recovery requests and progress-filter interventions."""

import csv
from collections import Counter
from unittest.mock import patch

from scripts.run_four_method_evaluation import run_episode
from src.evaluation.model_loader import load_validation_models
from src.safety.m7_controller import M7Controller


def main():
    with open("results/four_method_eval_matrix.csv", newline="") as f:
        matrix = list(csv.DictReader(f))

    spec = next(
        r for r in matrix
        if int(r["episode_id"]) == 10
        and int(r["latency_steps"]) == 4
        and float(r["immediate_probability"]) == 0.8
    )

    stats = Counter()
    original = M7Controller.select_action

    def observed(self, *args, **kwargs):
        stats["decisions"] += 1

        if kwargs.get("prefer_progress", False):
            stats["progress_requested"] += 1

        if self.enable_progress_forwarding:
            stats["forwarding_enabled"] += 1

        before = self.action_filter.intervention_count
        result = original(self, *args, **kwargs)
        after = self.action_filter.intervention_count

        if after > before:
            stats["filter_interventions"] += 1

        if kwargs.get("prefer_progress", False) and after > before:
            stats["interventions_during_recovery"] += 1

        return result

    models = load_validation_models(
        method="M7",
        agent_count=int(spec["agent_count"]),
        seed=int(spec["seed"]),
        episodes_per_scenario=20,
    )

    with patch.object(M7Controller, "select_action", observed):
        outcome = run_episode(
            method="DT-PROGRESS-M7",
            specification=spec,
            models=models,
            max_steps=400,
        )

    print("\nEPISODE 10 — PROGRESS RECOVERY")
    print("Success:", outcome["success"])
    print("Steps:", outcome["makespan"])

    for key in (
        "decisions",
        "progress_requested",
        "forwarding_enabled",
        "filter_interventions",
        "interventions_during_recovery",
    ):
        print(f"{key}: {stats[key]}")


if __name__ == "__main__":
    main()
