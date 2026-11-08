"""Check that experimental Hybrid M7 uses the DT observation builder."""

import csv
from unittest.mock import patch

import src.marl.real_training_step as training
from scripts.run_four_method_evaluation import run_episode
from src.evaluation.model_loader import load_validation_models

with open("results/dt_inferred_uncertain_matrix.csv", newline="") as f:
    rows = list(csv.DictReader(f))

spec = next(r for r in rows if int(r["episode_id"]) == 10)

original_dt_builder = training.build_dt_only_agent_inputs

for method in ("DT-ONLY-HYBRID", "DT-ROBUST-HYBRID"):
    counts = {"dt": 0}

    def counted_dt_builder(**kwargs):
        counts["dt"] += 1
        return original_dt_builder(**kwargs)

    models = load_validation_models(
        method="M7",
        agent_count=int(spec["agent_count"]),
        seed=int(spec["seed"]),
        episodes_per_scenario=20,
    )

    with (
        patch.object(
            training,
            "build_all_agent_inputs",
            side_effect=AssertionError(
                "Legacy simulator-position observations used!"
            ),
        ),
        patch.object(
            training,
            "build_dt_only_agent_inputs",
            side_effect=counted_dt_builder,
        ),
    ):
        result = run_episode(
            method, spec, models, max_steps=400
        )

    assert counts["dt"] > 0
    print(
        f"{method}: "
        f"DT calls={counts['dt']}, "
        f"success={result['success']}, "
        f"steps={result['makespan']}"
    )

print("PASS: Both Hybrid M7 methods used DT observations")
