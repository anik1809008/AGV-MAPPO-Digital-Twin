"""Diagnostic: verify M7 and robust progress execute inside Hybrid."""

import csv
from unittest.mock import patch

import src.marl.real_training_step as training
import src.safety.m7_controller as controller_module

from scripts.run_four_method_evaluation import run_episode
from src.evaluation.model_loader import load_validation_models


with open("results/dt_inferred_uncertain_matrix.csv", newline="") as f:
    specifications = list(csv.DictReader(f))

spec = next(
    row for row in specifications
    if int(row["episode_id"]) == 10
)

original_robust = controller_module.select_robust_progress_action
original_dt_observation = training.build_dt_only_agent_inputs

for method in ("DT-ONLY-HYBRID", "DT-ROBUST-HYBRID"):
    counts = {
        "robust_calls": 0,
        "robust_selected": 0,
        "dt_observation_calls": 0,
    }

    def counted_robust(**kwargs):
        counts["robust_calls"] += 1
        result = original_robust(**kwargs)
        if result is not None:
            counts["robust_selected"] += 1
        return result

    def counted_dt_observation(**kwargs):
        counts["dt_observation_calls"] += 1
        return original_dt_observation(**kwargs)

    models = load_validation_models(
        method="M7",
        agent_count=int(spec["agent_count"]),
        seed=int(spec["seed"]),
        episodes_per_scenario=20,
    )

    with (
        patch.object(
            controller_module,
            "select_robust_progress_action",
            side_effect=counted_robust,
        ),
        patch.object(
            training,
            "build_dt_only_agent_inputs",
            side_effect=counted_dt_observation,
        ),
    ):
        result = run_episode(
            method,
            spec,
            models,
            max_steps=400,
        )

    print(f"\n{method}")
    print("Success:", result["success"])
    print("Makespan:", result["makespan"])
    print("DT observation calls:", counts["dt_observation_calls"])
    print("Robust selector calls:", counts["robust_calls"])
    print("Robust actions selected:", counts["robust_selected"])
    print("Switch count:", result["switch_count"])

    assert counts["dt_observation_calls"] > 0, (
        "DT-only observations were not exercised"
    )

    if method == "DT-ONLY-HYBRID":
        assert counts["robust_calls"] == 0
    else:
        assert counts["robust_calls"] > 0
        assert counts["robust_selected"] > 0

print("\nPASS: Robust Progress activation verified")
