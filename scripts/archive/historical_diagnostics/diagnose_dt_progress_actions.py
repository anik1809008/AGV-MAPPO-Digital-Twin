"""Measure whether DT progress recovery changes filtered actions."""

import csv
from collections import Counter
from unittest.mock import patch

from scripts.run_four_method_evaluation import run_episode
from src.evaluation.model_loader import load_validation_models
from src.safety.m7_action_filter import M7ActionFilter


def main():
    with open("results/four_method_eval_matrix.csv", newline="") as f:
        matrix = list(csv.DictReader(f))

    spec = next(
        row for row in matrix
        if int(row["episode_id"]) == 10
        and int(row["latency_steps"]) == 4
        and float(row["immediate_probability"]) == 0.8
    )

    stats = Counter()
    by_agent = Counter()
    original = M7ActionFilter.select_safe_action

    def observed(self, *args, **kwargs):
        agent_id = kwargs["agent_id"]
        prefer = kwargs.get("prefer_progress", False)
        ranked = kwargs["ranked_actions"]
        original_action = ranked[0][0]
        position = kwargs.get("current_position")
        goal = kwargs.get("goal_position")

        if prefer:
            stats["progress_branch_eligible"] += (
                position is not None and goal is not None
            )
            stats["recovery_requests"] += 1
            by_agent[agent_id] += 1

            if position is not None and position == goal:
                stats["recovery_at_estimated_goal"] += 1

        result = original(self, *args, **kwargs)

        if prefer:
            if result[0] != original_action:
                stats["changed_top_action"] += 1
            else:
                stats["kept_top_action"] += 1

        return result

    models = load_validation_models(
        method="M7",
        agent_count=int(spec["agent_count"]),
        seed=int(spec["seed"]),
        episodes_per_scenario=20,
    )

    with patch.object(
        M7ActionFilter,
        "select_safe_action",
        observed,
    ):
        outcome = run_episode(
            method="DT-ESTIMATED-PROGRESS-M7",
            specification=spec,
            models=models,
            max_steps=400,
        )

    print("\nEPISODE 10 — ACTION SELECTION")
    print("Success:", outcome["success"])
    print("Steps:", outcome["makespan"])

    for key in (
        "recovery_requests",
        "progress_branch_eligible",
        "recovery_at_estimated_goal",
        "changed_top_action",
        "kept_top_action",
    ):
        print(f"{key}: {stats[key]}")

    print("\nRecovery requests by AGV:")
    for agent_id in sorted(by_agent):
        print(f"AGV {agent_id}: {by_agent[agent_id]}")


if __name__ == "__main__":
    main()
