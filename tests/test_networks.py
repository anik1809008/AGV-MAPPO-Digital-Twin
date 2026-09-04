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
