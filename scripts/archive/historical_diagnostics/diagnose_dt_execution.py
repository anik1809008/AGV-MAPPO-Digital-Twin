"""Diagnose command delivery and physical movement for episode 10."""

import csv
from collections import Counter
from unittest.mock import patch

import src.marl.real_training_step as training
from src.environment.delayed_execution import DelayedCommandExecutor
from scripts.run_four_method_evaluation import run_episode
from src.evaluation.model_loader import load_validation_models


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
    original_step = training.run_real_training_step
    original_ready = DelayedCommandExecutor.get_ready_actions

    def observed_ready(self, current_timestep):
        for queue in self.pending_commands.values():
            ready_count = 0

            # Match the executor's FIFO processing behavior.
            for execution_timestep, _ in queue:
                if execution_timestep > current_timestep:
                    break
                ready_count += 1

            stats["ready_commands"] += ready_count
            stats["superseded_commands"] += max(0, ready_count - 1)

        return original_ready(self, current_timestep)

    def observed_step(**kwargs):
        simulator = kwargs["simulator"]
        before = dict(simulator.agent_positions)

        result = original_step(**kwargs)

        for agent_id, previous in before.items():
            issued = int(result["actions"][agent_id])
            executed = int(result["executed_actions"][agent_id])
            moved = simulator.agent_positions[agent_id] != previous

            stats["agent_steps"] += 1
            stats["issued_moves"] += issued != 0
            stats["executed_moves"] += executed != 0
            stats["actual_moves"] += moved

            if issued != 0 and executed == 0:
                stats["issued_move_executed_wait"] += 1

            if executed != 0 and not moved:
                stats["executed_move_no_progress"] += 1

            if executed == 0 and moved:
                stats["movement_during_wait"] += 1

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
        side_effect=observed_step,
    ), patch.object(
        DelayedCommandExecutor,
        "get_ready_actions",
        observed_ready,
    ):
        outcome = run_episode(
            method="DT-INFERRED-M7",
            specification=spec,
            models=models,
            max_steps=400,
        )

    print("\nEPISODE 10 — EXECUTION DIAGNOSIS")
    print("Success:", outcome["success"])
    print("Steps:", outcome["makespan"])

    for name in (
        "agent_steps",
        "issued_moves",
        "executed_moves",
        "actual_moves",
        "issued_move_executed_wait",
        "executed_move_no_progress",
        "movement_during_wait",
        "ready_commands",
        "superseded_commands",
    ):
        print(f"{name}: {stats[name]}")


if __name__ == "__main__":
    main()
