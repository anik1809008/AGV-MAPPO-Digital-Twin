from src.environment.execution_delay import (
    ExecutionDelayModel,
)
from src.evaluation.context_factory import (
    build_experiment_context,
)
from src.evaluation.controller_factory import (
    build_method_controllers,
)
from src.evaluation.model_loader import (
    load_evaluation_models,
)
from src.evaluation.real_method_callback import (
    run_real_method,
)
from src.marl.scenario_split import (
    get_random_scenario_path,
)
from src.marl.training_instance import (
    build_training_instance,
)


def test_final_evaluation_m1_smoke():
    instance = build_training_instance(
        map_path=(
            "benchmarks/movingai/"
            "warehouse-10-20-10-2-1.map"
        ),
        scenario_path=get_random_scenario_path(21),
        agent_count=8,
        start_index=0,
    )

    grid = instance["grid"]
    starts = instance["starts"]
    goals = instance["goals"]

    models = load_evaluation_models(
        method="M1",
        agent_count=8,
        seed=0,
        scenario_id=21,
    )

    controllers = build_method_controllers(
        method="M1",
        actor=models["actor"],
        grid=grid,
    )

    context = build_experiment_context(
        grid=grid,
        starts=starts,
        goals=goals,
        actor=models["actor"],
        critic=models["critic"],
        latency_steps=0,
        max_steps=5,
        m4_controller=controllers[
            "m4_controller"
        ],
        m5_baseline=controllers[
            "m5_baseline"
        ],
        execution_delay_model=(
            ExecutionDelayModel(
                immediate_probability=1.0,
                seed=0,
            )
        ),
    )

    metrics = run_real_method(
        method="M1",
        latency_steps=0,
        context=context,
    )

    assert metrics["method"] == "M1"
    assert "success" in metrics
    assert "collision" in metrics
    assert "deadlock" in metrics
    assert "makespan" in metrics
    assert "path_length" in metrics
    assert "planning_time" in metrics
