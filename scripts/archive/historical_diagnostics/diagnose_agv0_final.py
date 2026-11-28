"""Diagnose AGV 0 near the end of the 800-step episode."""

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

    original = training.run_real_training_step
    stats = Counter()
    records = []

    def observed(**kwargs):
        sim = kwargs["simulator"]
        step = kwargs["current_timestep"]
        before = tuple(sim.agent_positions[0])
        goal = tuple(sim.agent_goals[0])
        possible = kwargs["possible_current_positions"][0]

        result = original(**kwargs)

        after = tuple(sim.agent_positions[0])
        action = int(result["executed_actions"][0])

        if step >= 700:
            distance = sim.navigation_distance(after, goal)
            stats["steps"] += 1
            stats["movement_commands"] += action != 0
            stats["actual_movements"] += after != before
            stats["geometry_blocked"] += (
                action != 0
                and before != goal
                and sim.get_next_position(before, action) == before
            )
            stats["repeated_position"] += after == before

            records.append((
                step, before, after, action,
                distance, len(possible),
            ))

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
            max_steps=800,
        )

    print("Success:", outcome["success"])
    print("Steps:", outcome["makespan"])
    print("AGV 0 final:", records[-1][2] if records else None)

    for key, value in stats.items():
        print(f"{key}: {value}")

    print("\nLast 20 steps:")
    for step, before, after, action, distance, size in records[-20:]:
        print(
            f"step={step} {before}->{after} "
            f"action={action} path_distance={distance} "
            f"DT_set_size={size}"
        )


if __name__ == "__main__":
    main()
