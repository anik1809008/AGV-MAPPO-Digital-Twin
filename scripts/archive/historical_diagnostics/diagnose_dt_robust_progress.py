"""Check availability of goal-improving actions across possible DT positions."""

import csv
from collections import Counter
from unittest.mock import patch

import src.marl.real_training_step as training
from scripts.run_four_method_evaluation import run_episode
from src.evaluation.model_loader import load_validation_models
from src.environment.actions import Action
from src.environment.pathfinding import shortest_path_distance
from src.safety.shield import SafetyShield


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
        sim = kwargs["simulator"]
        step = kwargs["current_timestep"]
        possible = kwargs["possible_current_positions"]
        shield = SafetyShield(sim.grid)

        if step >= 700:
            agent_id = 0
            goal = sim.agent_goals[agent_id]
            candidates = set(possible[agent_id])

            improving = []

            for action in Action:
                if action == Action.WAIT:
                    continue

                changes = []

                for position in candidates:
                    before = shortest_path_distance(
                        sim.grid, position, goal
                    )
                    after_position = shield.next_position(
                        position, action
                    )
                    after = shortest_path_distance(
                        sim.grid, after_position, goal
                    )

                    if before is None or after is None:
                        break

                    changes.append(after - before)

                if len(changes) == len(candidates):
                    if all(change < 0 for change in changes):
                        improving.append(action)

            stats["checks"] += 1
            stats["robust_improving_available"] += bool(improving)
            stats["max_set_size"] = max(
                stats["max_set_size"], len(candidates)
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
            method="DT-INFERRED-M7",
            specification=spec,
            models=models,
            max_steps=800,
        )

    print("\nAGV 0 — FINAL 100 STEPS")
    print("Success:", outcome["success"])
    print("Checks:", stats["checks"])
    print(
        "Steps with robust improving action:",
        stats["robust_improving_available"],
    )
    print("Maximum uncertainty size:", stats["max_set_size"])


if __name__ == "__main__":
    main()
