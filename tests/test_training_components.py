import torch

from src.marl.training_components import (
    OBSERVATION_DIM,
    build_mappo_components,
)


def test_build_components_for_eight_agents():
    components = build_mappo_components(
        agent_count=8,
    )

    actor = components["actor"]
    critic = components["critic"]

    actor_output = actor(
        torch.zeros(1, OBSERVATION_DIM)
    )

    critic_output = critic(
        torch.zeros(
            1,
            OBSERVATION_DIM * 8,
        )
    )

    assert actor_output.shape == (1, 5)
    assert critic_output.shape == (1, 1)


def test_build_components_for_twenty_agents():
    components = build_mappo_components(
        agent_count=20,
    )

    actor = components["actor"]
    critic = components["critic"]

    actor_output = actor(
        torch.zeros(1, OBSERVATION_DIM)
    )

    critic_output = critic(
        torch.zeros(
            1,
            OBSERVATION_DIM * 20,
        )
    )

    assert actor_output.shape == (1, 5)
    assert critic_output.shape == (1, 1)


def test_invalid_agent_count_raises_error():
    try:
        build_mappo_components(
            agent_count=0,
        )
    except ValueError as exc:
        assert str(exc) == (
            "agent_count must be >= 1"
        )
    else:
        raise AssertionError(
            "Expected ValueError"
        )
