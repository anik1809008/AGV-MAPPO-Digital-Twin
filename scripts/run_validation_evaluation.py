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
        "--method",
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
        type=int,
        choices=[8, 20],
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--scenario-id",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--episodes-per-scenario",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--latency",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--immediate-probability",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--m5-threshold",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
    )

    parser.add_argument(
        "--results-path",
        type=str,
        default="results/validation_evaluation.csv",
    )

    args = parser.parse_args()

    if (
        args.method != "M1"
        and args.episodes_per_scenario is None
    ):
        raise ValueError(
            "--episodes-per-scenario is required "
            "for learned validation methods"
        )

    if (
        args.episodes_per_scenario is not None
        and args.episodes_per_scenario < 1
    ):
        raise ValueError(
            "--episodes-per-scenario must be >= 1"
        )

    if args.scenario_id not in VALIDATION_SCENARIO_IDS:
        raise ValueError(
            "Validation scenario_id must be "
            "in the validation split: 16 to 20"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    scenario_path = get_random_scenario_path(
        args.scenario_id
    )

    instance = build_training_instance(
        map_path=MAP_PATH,
        scenario_path=scenario_path,
        agent_count=args.agents,
        start_index=args.start_index,
    )

    grid = instance["grid"]
    starts = instance["starts"]
    goals = instance["goals"]

    if args.method == "M1":
        validation_budget = 1
    else:
        validation_budget = args.episodes_per_scenario

    models = load_validation_models(
        method=args.method,
        agent_count=args.agents,
        seed=args.seed,
        episodes_per_scenario=validation_budget,
    )

    controllers = build_method_controllers(
        method=args.method,
        actor=models["actor"],
        grid=grid,
        m5_threshold=args.m5_threshold,
    )

    execution_delay_model = ExecutionDelayModel(
        immediate_probability=(
            args.immediate_probability
        ),
        seed=args.seed,
    )

    context = build_experiment_context(
        grid=grid,
        starts=starts,
        goals=goals,
        actor=models["actor"],
        critic=models["critic"],
        latency_steps=args.latency,
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
        method=args.method,
        latency_steps=args.latency,
        context=context,
    )

    metrics["scenario_id"] = args.scenario_id
    metrics["agent_count"] = args.agents
    metrics["seed"] = args.seed
    metrics["checkpoint_scenario_id"] = ""
    metrics["episodes_per_scenario"] = (
        ""
        if args.method == "M1"
        else args.episodes_per_scenario
    )
    metrics["latency_steps"] = args.latency
    metrics["immediate_probability"] = (
        args.immediate_probability
    )
    metrics["uncertainty_condition"] = "manual"

    if args.method == "M5":
        metrics["m5_threshold"] = (
            args.m5_threshold
        )
    else:
        metrics["m5_threshold"] = ""

    append_result_csv(
        args.results_path,
        metrics,
    )

    print(metrics)

    if models["checkpoint_path"] is not None:
        print(
            "Validation checkpoint:",
            models["checkpoint_path"],
        )


if __name__ == "__main__":
    main()
