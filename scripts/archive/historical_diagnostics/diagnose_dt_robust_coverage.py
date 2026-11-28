"""Measure DT truth coverage and uncertainty under probabilistic execution."""

import csv
from collections import Counter
from statistics import mean
from unittest.mock import patch

import src.marl.episode_runner as runner
from scripts.run_four_method_evaluation import run_episode
from src.evaluation.model_loader import load_validation_models


def evaluate(spec):
    stats = Counter()
    sizes = []
    misses = []
    simulator_ref = None

    original_inputs = runner.get_digital_twin_inputs
    original_episode = runner.run_episode

    def observed(*args, **kwargs):
        result = original_inputs(*args, **kwargs)

        for agent_id in kwargs["digital_twin"].agent_states:
            truth = tuple(simulator_ref.agent_positions[agent_id])
            possible = set(result["possible_current_positions"][agent_id])

            stats["checks"] += 1
            stats["covered"] += truth in possible
            stats["singleton"] += len(possible) == 1
            stats["singleton_correct"] += len(possible) == 1 and truth in possible

            if len(possible) > 1:
                trusted = tuple(result["trusted_positions"][agent_id])
                distance = (
                    abs(trusted[0] - truth[0])
                    + abs(trusted[1] - truth[1])
                )
                stats["multi_checks"] += 1
                stats["stale_fallback"] += distance > 0
                stats["fallback_distance_total"] += distance
                stats["fallback_distance_max"] = max(
                    stats["fallback_distance_max"], distance
                )

            sizes.append(len(possible))

            if truth not in possible and len(misses) < 5:
                misses.append((
                    kwargs["current_timestep"],
                    agent_id,
                    truth,
                    len(possible),
                ))

        return result

    def wrapped_episode(**kwargs):
        nonlocal simulator_ref
        simulator_ref = kwargs["simulator"]
        return original_episode(**kwargs)

    models = load_validation_models(
        method="M7",
        agent_count=int(spec["agent_count"]),
        seed=int(spec["seed"]),
        episodes_per_scenario=20,
    )

    with patch.object(runner, "get_digital_twin_inputs", observed), \
         patch(
             "src.evaluation.marl_method_runner.run_episode",
             side_effect=wrapped_episode,
         ):
        outcome = run_episode(
            method="DT-ROBUST-PROGRESS-M7",
            specification=spec,
            models=models,
            max_steps=400,
        )

    checks = max(stats["checks"], 1)

    print(
        f"\nProbability={spec['immediate_probability']} "
        f"Latency={spec['latency_steps']} "
        f"Episode={spec['episode_id']}"
    )
    print("Success:", outcome["success"])
    print("Position checks:", stats["checks"])
    print("Truth coverage (%):", round(100 * stats["covered"] / checks, 2))
    print("Singleton rate (%):", round(100 * stats["singleton"] / checks, 2))
    print("Correct singleton rate (%):", round(
        100 * stats["singleton_correct"] / max(stats["singleton"], 1), 2
    ))
    print("Mean uncertainty size:", round(mean(sizes), 2) if sizes else 0)
    print("Maximum uncertainty size:", max(sizes, default=0))
    multi = max(stats["multi_checks"], 1)
    print("Multi-position checks:", stats["multi_checks"])
    print("Stale fallback rate (%):", round(
        100 * stats["stale_fallback"] / multi, 2
    ))
    print("Mean fallback distance:", round(
        stats["fallback_distance_total"] / multi, 2
    ))
    print("Maximum fallback distance:", stats["fallback_distance_max"])
    print("First coverage misses:", misses)


def main():
    with open("results/four_method_eval_matrix.csv", newline="") as f:
        matrix = list(csv.DictReader(f))

    for probability in (1.0, 0.8):
        for latency in (0, 4):
            spec = next(
                row for row in matrix
                if int(row["latency_steps"]) == latency
                and float(row["immediate_probability"]) == probability
            )
            evaluate(spec)


if __name__ == "__main__":
    main()
