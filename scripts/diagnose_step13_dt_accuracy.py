
"""Verify same-timestep DT position inference."""

import json
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import scripts.run_four_method_evaluation as evaluation
import src.hybrid.two_way_episode_runner as hybrid

from scripts.diagnose_step13_failures import (
    load_csv,
    diagnose,
    MATRIX,
    REFERENCE,
)

from src.evaluation.model_loader import (
    load_validation_models,
)


TARGETS = (
    20131,
    20132,
    20138,
    20186,
)

OUTPUT = Path(
    "results/step13_dt_accuracy.jsonl"
)


def verify_episode(spec, reference, models):
    original_runner = evaluation.run_two_way_episode
    original_dt = hybrid.get_digital_twin_inputs

    captured = {}
    stats = Counter()
    mismatch_examples = []

    def observe_runner(*args, **kwargs):
        captured["context"] = kwargs["context"]

        return original_runner(*args, **kwargs)

    def observe_dt(*args, **kwargs):
        dt = original_dt(*args, **kwargs)

        context = captured["context"]
        sim = context["simulator"]
        executor = context["delayed_executor"]

        step = kwargs["current_timestep"]

        truth = dict(sim.agent_positions)
        candidates = dt["possible_current_positions"]
        ages = dt["aoi_values"]

        stats["timesteps"] += 1

        pending = any(
            bool(queue)
            for queue in executor.pending_commands.values()
        )

        complete = (
            set(truth)
            == set(candidates)
            == set(ages)
        )

        if not complete:
            stats["incomplete_dt_states"] += 1

        all_singleton = (
            complete
            and bool(truth)
            and all(
                len(candidates[a]) == 1
                for a in truth
            )
        )

        if all_singleton:
            stats["all_singleton_steps"] += 1

        if not pending:
            stats["empty_queue_steps"] += 1

        eligible = (
            all_singleton
            and not pending
            and all(
                0 <= age <= 2
                for age in ages.values()
            )
            and getattr(
                executor.delay_model,
                "immediate_probability",
                0.0,
            ) == 1.0
        )

        if eligible:
            stats["eligible_state_steps"] += 1

        all_correct = complete

        for agent_id, position in truth.items():
            positions = candidates.get(
                agent_id, set()
            )

            stats["agent_checks"] += 1

            covered = position in positions

            if covered:
                stats["truth_covered"] += 1
            else:
                stats["coverage_failures"] += 1
                all_correct = False

            if len(positions) == 1:
                stats["singleton_checks"] += 1

                if covered:
                    stats["correct_singletons"] += 1
                else:
                    stats["incorrect_singletons"] += 1

            if not covered and len(mismatch_examples) < 10:
                mismatch_examples.append({
                    "step": step,
                    "agent_id": int(agent_id),
                    "truth": list(position),
                    "candidates": [
                        list(p) for p in positions
                    ],
                    "aoi": ages.get(agent_id),
                    "pending": pending,
                })

        if all_correct:
            stats["all_truth_covered_steps"] += 1

        if eligible and not all_correct:
            stats["eligible_state_errors"] += 1

        if eligible and all_correct:
            stats["correct_eligible_states"] += 1

        return dt

    with patch.object(
        evaluation,
        "run_two_way_episode",
        side_effect=observe_runner,
    ), patch.object(
        hybrid,
        "get_digital_twin_inputs",
        side_effect=observe_dt,
    ):
        replay = diagnose(
            spec,
            reference,
            models,
        )

    checks = stats["agent_checks"]
    singleton_checks = stats["singleton_checks"]

    return {
        "episode_id": int(spec["episode_id"]),
        "latency": int(spec["latency_steps"]),
        "replay_matches_reference": replay[
            "replay_matches_reference"
        ],
        "replay_mismatches": replay[
            "replay_mismatches"
        ],
        "coverage_rate": (
            stats["truth_covered"] / checks
            if checks else None
        ),
        "singleton_accuracy": (
            stats["correct_singletons"]
            / singleton_checks
            if singleton_checks else None
        ),
        "statistics": dict(stats),
        "mismatch_examples": mismatch_examples,
    }


def main():
    matrix = {
        int(r["episode_id"]): r
        for r in load_csv(MATRIX)
    }

    reference = {
        int(r["episode_id"]): r
        for r in load_csv(REFERENCE)
    }

    total = Counter()
    matched = 0

    print("STEP 13 DT ACCURACY VALIDATION")

    with OUTPUT.open("w") as handle:
        for episode_id in TARGETS:
            spec = matrix[episode_id]

            models = load_validation_models(
                method="M7",
                agent_count=int(
                    spec["agent_count"]
                ),
                seed=int(spec["seed"]),
                episodes_per_scenario=20,
            )

            print(
                f"\nEpisode {episode_id}",
                flush=True,
            )

            record = verify_episode(
                spec,
                reference[episode_id],
                models,
            )

            stats = Counter(
                record["statistics"]
            )

            print(
                "Replay matches:",
                record["replay_matches_reference"],
            )

            print(
                "Replay mismatches:",
                record["replay_mismatches"],
            )

            print(
                "Timesteps:",
                stats["timesteps"],
            )

            print(
                "Agent checks:",
                stats["agent_checks"],
            )

            print(
                "Coverage rate:",
                record["coverage_rate"],
            )

            print(
                "Singleton accuracy:",
                record["singleton_accuracy"],
            )

            print(
                "Incorrect singletons:",
                stats["incorrect_singletons"],
            )

            print(
                "Eligible states:",
                stats["eligible_state_steps"],
            )

            print(
                "Eligible state errors:",
                stats["eligible_state_errors"],
            )

            if record["mismatch_examples"]:
                print(
                    "Mismatch examples:",
                    record["mismatch_examples"],
                )

            if record["replay_matches_reference"]:
                matched += 1
                total.update(stats)

            handle.write(
                json.dumps(record) + "\n"
            )
            handle.flush()

    print("\n=== MATCHED VALIDATION SUMMARY ===")

    print("Matched episodes:", matched)
    print("Total timesteps:", total["timesteps"])
    print("Total agent checks:", total["agent_checks"])

    print(
        "Coverage failures:",
        total["coverage_failures"],
    )

    print(
        "Incorrect singletons:",
        total["incorrect_singletons"],
    )

    print(
        "Eligible state steps:",
        total["eligible_state_steps"],
    )

    print(
        "Eligible state errors:",
        total["eligible_state_errors"],
    )

    print(
        "Correct eligible states:",
        total["correct_eligible_states"],
    )

    passed = (
        matched == len(TARGETS)
        and total["agent_checks"] > 0
        and total["eligible_state_steps"] > 0
        and total["incomplete_dt_states"] == 0
        and total["coverage_failures"] == 0
        and total["incorrect_singletons"] == 0
        and total["eligible_state_errors"] == 0
    )

    print("Pilot validation passed:", passed)
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
