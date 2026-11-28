"""Compare actual M7 choices with robust safe progress actions."""

import csv
from collections import Counter
from unittest.mock import patch

from scripts.run_four_method_evaluation import run_episode
from src.evaluation.model_loader import load_validation_models
from src.environment.actions import Action
from src.environment.pathfinding import shortest_path_distance
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
        result = original(self, *args, **kwargs)

        if kwargs["agent_id"] != 0:
            return result

        stats["decisions"] += 1

        goal = kwargs["goal_position"]
        possible = set(kwargs["possible_current_positions"])
        shield = self.shield

        robust_safe = set()

        for action in Action:
            if action == Action.WAIT:
                continue

            improves = True

            for position in possible:
                before = shortest_path_distance(
                    grid=shield.grid,
                    start=position,
                    goal=goal,
                )
                after = shortest_path_distance(
                    grid=shield.grid,
                    start=shield.next_position(position, action),
                    goal=goal,
                )

                if before is None or after is None or after >= before:
                    improves = False
                    break

            if not improves:
                continue

            if shield.is_action_safe(
                agent_id=0,
                possible_current_positions=possible,
                action=action,
                other_current_positions=kwargs["other_current_positions"],
                other_next_positions=kwargs["other_next_positions"],
                reachable_occupancies=kwargs["reachable_occupancies"],
                other_possible_transitions=kwargs.get(
                    "other_possible_transitions"
                ),
            ):
                robust_safe.add(action)

        if robust_safe:
            stats["safe_progress_available"] += 1
            stats["selected_safe_progress"] += (
                Action(result[0]) in robust_safe
            )
            stats["selected_other"] += (
                Action(result[0]) not in robust_safe
            )

        return result

    models = load_validation_models(
        method="M7",
        agent_count=int(spec["agent_count"]),
        seed=int(spec["seed"]),
        episodes_per_scenario=20,
    )

    with patch.object(M7Controller, "select_action", observed):
        outcome = run_episode(
            method="DT-INFERRED-M7",
            specification=spec,
            models=models,
            max_steps=800,
        )

    print("\nAGV 0 — CONTROLLER ACTION CHOICES")
    print("Success:", outcome["success"])
    print("Steps:", outcome["makespan"])

    for key in (
        "decisions",
        "safe_progress_available",
        "selected_safe_progress",
        "selected_other",
    ):
        print(f"{key}: {stats[key]}")

    available = stats["safe_progress_available"]
    if available:
        print(
            "Selected safe progress (%):",
            round(100 * stats["selected_safe_progress"] / available, 2),
        )


if __name__ == "__main__":
    main()
