"""Classify executed movement commands that do not move an AGV."""

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

    stats = Counter()
    original = training.run_real_training_step

    def observed(**kwargs):
        simulator = kwargs["simulator"]
        before = dict(simulator.agent_positions)
        goals = dict(simulator.agent_goals)

        result = original(**kwargs)
        after = dict(simulator.agent_positions)

        for agent_id, start in before.items():
            action = int(result["executed_actions"][agent_id])

            if action == 0:
                continue

            stats["executed_moves"] += 1

            if after[agent_id] != start:
                stats["moved"] += 1
            elif start == goals[agent_id]:
                stats["already_at_goal"] += 1
            elif simulator.get_next_position(start, action) == start:
                stats["blocked_by_geometry"] += 1
            elif result["collision"]:
                stats["joint_collision_rejected"] += 1
            else:
                stats["unexplained"] += 1

        return result

    models = load_validation_models(
        method="M7",
        agent_count=int(spec["agent_count"]),
        seed=int(spec["seed"]),
        episodes_per_scenario=20,
    )

    with patch.object(
        training,
        "run_real_training_step",
        side_effect=observed,
    ):
        outcome = run_episode(
            method="DT-INFERRED-M7",
            specification=spec,
            models=models,
            max_steps=400,
        )

    print("\nEPISODE 10 — MOVEMENT BLOCKERS")
    print("Success:", outcome["success"])
    print("Steps:", outcome["makespan"])

    for key in (
        "executed_moves",
        "moved",
        "already_at_goal",
        "blocked_by_geometry",
        "joint_collision_rejected",
        "unexplained",
    ):
        print(f"{key}: {stats[key]}")

    assert (
        stats["executed_moves"]
        == stats["moved"]
        + stats["already_at_goal"]
        + stats["blocked_by_geometry"]
        + stats["joint_collision_rejected"]
        + stats["unexplained"]
    )


if __name__ == "__main__":
    main()
