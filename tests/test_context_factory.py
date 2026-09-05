from src.evaluation.context_factory import build_experiment_context
from src.marl.networks import ActorNetwork, CriticNetwork


def test_build_experiment_context():
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
        max_steps=10,
    )

    assert context["simulator"].agent_positions == starts
    assert list(context["digital_twin"].agent_states.keys()) == [0, 1]
    assert context["buffer"].num_agents == 2
    assert context["telemetry_channel"].latency_steps == 2
    assert context["max_steps"] == 10
