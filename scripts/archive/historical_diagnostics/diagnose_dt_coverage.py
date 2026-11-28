"""Check whether DT uncertainty contains simulator ground truth."""

import csv
from collections import Counter
from statistics import mean
from unittest.mock import patch

from scripts.run_four_method_evaluation import run_episode
from src.evaluation.model_loader import load_validation_models
import src.marl.episode_runner as runner


def main():
    with open("results/four_method_eval_matrix.csv", newline="") as f:
        matrix = list(csv.DictReader(f))

    original = runner.get_digital_twin_inputs

    for latency in (0, 4):
        spec = next(
            row for row in matrix
            if int(row["latency_steps"]) == latency
            and float(row["immediate_probability"]) == 1.0
        )

        stats = Counter()
        sizes = []
        mismatch_examples = []
        simulator_ref = None

        def observed(*args, **kwargs):
            result = original(*args, **kwargs)

            for agent_id, state in kwargs["digital_twin"].agent_states.items():
                true_position = simulator_ref.agent_positions[agent_id]
                possible = result["possible_current_positions"][agent_id]
                trusted = result["trusted_positions"][agent_id]

                stats["checks"] += 1
                stats["covered"] += true_position in possible
                sizes.append(len(possible))

                if true_position not in possible and len(mismatch_examples) < 5:
                    mismatch_examples.append((
                        kwargs["current_timestep"],
                        agent_id,
                        true_position,
                        trusted,
                        len(possible),
                        result["aoi_values"][agent_id],
                    ))
            return result

        models = load_validation_models(
            method="M7",
            agent_count=int(spec["agent_count"]),
            seed=int(spec["seed"]),
            episodes_per_scenario=20,
        )

        # Capture the simulator passed to the actual episode runner.
        original_episode = runner.run_episode

        def wrapped_episode(**kwargs):
            nonlocal simulator_ref
            simulator_ref = kwargs["simulator"]
            return original_episode(**kwargs)

        with patch.object(runner, "get_digital_twin_inputs", observed), \
             patch("src.evaluation.marl_method_runner.run_episode",
                   side_effect=wrapped_episode):

            outcome = run_episode(
                method="DT-OBSERVED-M7",
                specification=spec,
                models=models,
                max_steps=400,
            )

        print(f"\nLatency {latency}, episode {spec['episode_id']}")
        print("Success:", outcome["success"])
        print("Position checks:", stats["checks"])
        print("Truth covered:", stats["covered"])
        print("Coverage (%):", round(
            100 * stats["covered"] / max(stats["checks"], 1), 2
        ))
        print("Mean uncertainty size:", round(mean(sizes), 2))
        print("Maximum uncertainty size:", max(sizes))
        print("First misses (step, agent, truth, trusted, size, AoI):")
        for item in mismatch_examples:
            print(item)


if __name__ == "__main__":
    main()
