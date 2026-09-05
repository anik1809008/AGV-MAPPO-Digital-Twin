import torch

from src.marl.networks import ActorNetwork, CriticNetwork


def test_actor_output_shape():
    input_dim = 3 * 9 * 9 + 4

    actor = ActorNetwork(
        input_dim=input_dim,
        action_dim=5,
    )

    x = torch.zeros((2, input_dim))

    output = actor(x)

    assert output.shape == (2, 5)


def test_critic_output_shape():
    input_dim = 3 * 9 * 9 + 4

    critic = CriticNetwork(
        input_dim=input_dim,
    )

    x = torch.zeros((2, input_dim))

    output = critic(x)

    assert output.shape == (2, 1)
import numpy as np

from src.marl.centralized_state import build_centralized_state


def test_centralized_state_shape_for_eight_agents():
    observations = [
        np.zeros(247, dtype=np.float32)
        for _ in range(8)
    ]

    state = build_centralized_state(observations)

    assert state.shape == (1976,)


def test_critic_accepts_centralized_state():
    critic = CriticNetwork(
        input_dim=1976,
    )

    state = torch.zeros(
        1,
        1976,
        dtype=torch.float32,
    )

    value = critic(state)

    assert value.shape == (1, 1)
