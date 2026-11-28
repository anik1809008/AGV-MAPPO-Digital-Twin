"""Check safe, goal-improving actions for AGV 0 in episode 10."""

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
        simulator = kwargs["simulator"]
        step = kwargs["current_timestep"]

        if step >= 700:
            agent_id = 0
            goal = simulator.agent_goals[agent_id]
            possible = set(
                kwargs["possible_current_positions"][agent_id]
            )
            shield = SafetyShield(simulator.grid)

            robust_actions = []
            safe_robust_actions = []

            for action in Action:
                if action == Action.WAIT:
                    continue

                improves_everywhere = True

                for position in possible:
                    next_position = shield.next_position(position, action)

                    before = shortest_path_distance(
                        simulator.grid, position, goal
                    )
                    after = shortest_path_distance(
                        simulator.grid, next_position, goal
                    )

                    if (
                        before is None
                        or after is None
                        or after >= before
                    ):
                        improves_everywhere = False
                        break

                if not improves_everywhere:
                    continue

                robust_actions.append(action)

                is_safe = shield.is_action_safe(
                    agent_id=agent_id,
                    possible_current_positions=possible,
                    action=action,
                    other_current_positions={
                        other_id: position
                        for other_id, position
                        in kwargs["trusted_positions"].items()
                        if other_id != agent_id
                    },
                    other_next_positions={},
                    reachable_occupancies=kwargs[
                        "reachable_occupancies"
                    ],
                    other_possible_transitions=(
                        kwargs.get("possible_transitions") or {}
                    ),
                )

                if is_safe:
                    safe_robust_actions.append(action)

            stats["checks"] += 1
            stats["robust_available"] += bool(robust_actions)
            stats["safe_robust_available"] += bool(
                safe_robust_actions
            )

        result = original(**kwargs)

        if step >= 700:
            stats["issued_moves"] += (
                int(result["actions"][0]) != int(Action.WAIT)
            )
            stats["executed_moves"] += (
                int(result["executed_actions"][0]) != int(Action.WAIT)
            )

        return result

    models = load_validation_models(
        method="M7",
        agent_count=int(spec["agent_count"]),
        seed=int(spec["seed"]),
        episodes_per_scenario=20,
    )

    with patch.object(
        training, "run_real_training_step", side_effect=observed
    ):
        outcome = run_episode(
            method="DT-INFERRED-M7",
            specification=spec,
            models=models,
            max_steps=800,
        )

    print("\nAGV 0 — FINAL 100 STEPS")
    print("Success:", outcome["success"])
    print("Steps checked:", stats["checks"])
    print("Robust progress available:", stats["robust_available"])
    print("Shield-safe robust progress available:",
          stats["safe_robust_available"])
    print("Issued movement commands:", stats["issued_moves"])
    print("Executed movement commands:", stats["executed_moves"])


if __name__ == "__main__":
    main()
