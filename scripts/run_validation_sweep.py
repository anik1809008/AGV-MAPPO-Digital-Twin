import argparse
import random

import numpy as np
import torch

from src.environment.execution_delay import ExecutionDelayModel
from src.evaluation.context_factory import build_experiment_context
from src.evaluation.controller_factory import (
    build_method_controllers,
)
from src.evaluation.experiment_config import (
    DEFAULT_MAX_STEPS,
    M5_THRESHOLDS,
    build_validation_matrix,
)
from src.evaluation.model_loader import (
    load_validation_models,
)
from src.evaluation.real_method_callback import (
    run_real_method,
)
from src.evaluation.results_writer import (
    append_result_csv,
)
from src.marl.scenario_split import (
    VALIDATION_SCENARIO_IDS,
    get_random_scenario_path,
)
from src.marl.training_instance import (
    build_training_instance,
)


MAP_PATH = (
    "benchmarks/movingai/"
    "warehouse-10-20-10-2-1.map"
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--methods",
        nargs="+",
        choices=[
            "M1",
            "M2",
            "M3",
            "M4",
            "M5",
            "M6",
        ],
        required=True,
    )

    parser.add_argument(
        "--agents",
        nargs="+",
        type=int,
        choices=[8, 20],
        required=True,
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--training-budgets",
        nargs="+",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
    )

    parser.add_argument(
        "--results-path",
        type=str,
        default="results/validation_sweep.csv",
    )

    args = parser.parse_args()

    if any(
        budget < 1
        for budget in args.training_budgets
    ):
        raise ValueError(
            "All training budgets must be >= 1"
        )

    matrix = build_validation_matrix(
        methods=args.methods,
        agent_counts=args.agents,
        seeds=args.seeds,
        scenario_ids=VALIDATION_SCENARIO_IDS,
        training_budgets=args.training_budgets,
    )

    run_count = 0

    for row in matrix:
        method = row["method"]
        agent_count = row["agent_count"]
        seed = row["seed"]
        scenario_id = row["scenario_id"]
        episodes_per_scenario = row[
            "episodes_per_scenario"
        ]
        latency = row["latency"]
        immediate_probability = row[
            "immediate_probability"
        ]
        uncertainty_condition = row[
            "uncertainty_condition"
        ]

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        scenario_path = get_random_scenario_path(
            scenario_id
        )

        instance = build_training_instance(
            map_path=MAP_PATH,
            scenario_path=scenario_path,
            agent_count=agent_count,
            start_index=args.start_index,
        )

        grid = instance["grid"]
        starts = instance["starts"]
        goals = instance["goals"]

        if method == "M1":
            validation_budget = 1
        else:
            validation_budget = (
                episodes_per_scenario
            )

        models = load_validation_models(
            method=method,
            agent_count=agent_count,
            seed=seed,
            episodes_per_scenario=validation_budget,
        )

        if method == "M5":
            thresholds = M5_THRESHOLDS
        else:
            thresholds = [None]

        for m5_threshold in thresholds:
            controllers = build_method_controllers(
                method=method,
                actor=models["actor"],
                grid=grid,
                m5_threshold=(
                    2
                    if m5_threshold is None
                    else m5_threshold
                ),
            )

            execution_delay_model = (
                ExecutionDelayModel(
                    immediate_probability=(
                        immediate_probability
                    ),
                    seed=seed,
                )
            )

            context = build_experiment_context(
                grid=grid,
                starts=starts,
                goals=goals,
                actor=models["actor"],
                critic=models["critic"],
                latency_steps=latency,
                max_steps=args.max_steps,
                m4_controller=controllers[
                    "m4_controller"
                ],
                m5_baseline=controllers[
                    "m5_baseline"
                ],
                execution_delay_model=(
                    execution_delay_model
                ),
            )

            metrics = run_real_method(
                method=method,
                latency_steps=latency,
                context=context,
            )

            metrics["scenario_id"] = scenario_id
            metrics["agent_count"] = agent_count
            metrics["seed"] = seed
            metrics["checkpoint_scenario_id"] = ""
            metrics["episodes_per_scenario"] = (
                ""
                if method == "M1"
                else episodes_per_scenario
            )
            metrics["uncertainty_condition"] = (
                uncertainty_condition
            )
            metrics["latency_steps"] = latency
            metrics["immediate_probability"] = (
                immediate_probability
            )

            if method == "M5":
                metrics["m5_threshold"] = (
                    m5_threshold
                )
            else:
                metrics["m5_threshold"] = ""

            append_result_csv(
                args.results_path,
                metrics,
            )

            run_count += 1

            print(
                f"Run {run_count}: "
                f"method={method}, "
                f"agents={agent_count}, "
                f"seed={seed}, "
                f"scenario={scenario_id}, "
                f"budget={episodes_per_scenario}, "
                f"condition={uncertainty_condition}, "
                f"latency={latency}, "
                f"p={immediate_probability}, "
                f"m5_threshold={m5_threshold}"
            )

    print(
        "Validation sweep complete. "
        f"Total runs: {run_count}"
    )


if __name__ == "__main__":
    main()
