from src.evaluation.context_factory import (
    build_experiment_context,
)
from src.evaluation.real_method_callback import (
    run_real_method,
)
from src.marl.networks import ActorNetwork, CriticNetwork


def test_real_method_callback_imports():
    assert callable(run_real_method)


def test_real_method_callback_runs_m6():
    grid = [
        [0, 0, 0],
        [0, 0, 0],
    ]

    starts = {
        0: (0, 0),
        1: (2, 1),
    }

    goals = {
        0: (2, 0),
        1: (0, 1),
    }

    actor = ActorNetwork(
        input_dim=247,
        action_dim=5,
    )

    critic = CriticNetwork(
        input_dim=494,
    )

    context = build_experiment_context(
        grid=grid,
        starts=starts,
        goals=goals,
        actor=actor,
        critic=critic,
        latency_steps=2,
        max_steps=4,
    )

    metrics = run_real_method(
        method="M6",
        latency_steps=2,
        context=context,
    )

    assert metrics["method"] == "M6"
    assert "success" in metrics
    assert "collision" in metrics
    assert "deadlock" in metrics
    assert "makespan" in metrics
    assert "path_length" in metrics
