"""Inspect distance-to-goal bounds under DT position uncertainty."""

import csv
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
    best_lower = {0: None, 1: None}

    def observed(**kwargs):
        step = kwargs["current_timestep"]
        simulator = kwargs["simulator"]
        possible = kwargs["possible_current_positions"]

        for agent_id in (0, 1):
            goal = simulator.agent_goals[agent_id]
            positions = set(possible[agent_id])

            distances = [
                abs(p[0] - goal[0]) + abs(p[1] - goal[1])
                for p in positions
            ]

            lower = min(distances)
            upper = max(distances)

            previous = best_lower[agent_id]
            if previous is None or lower < previous:
                best_lower[agent_id] = lower

            if step % 50 == 0 or step == 399:
                print(
                    f"step={step:3d} AGV={agent_id} "
                    f"set={len(positions):2d} "
                    f"min_distance={lower:3d} "
                    f"max_distance={upper:3d} "
                    f"best_min={best_lower[agent_id]:3d} "
                    f"goal_possible={goal in positions}"
                )

        return original(**kwargs)

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
            method="DT-ESTIMATED-PROGRESS-M7",
            specification=spec,
            models=models,
            max_steps=400,
        )

    print("Success:", outcome["success"])
    print("Steps:", outcome["makespan"])


if __name__ == "__main__":
    main()
