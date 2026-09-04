import torch
from torch.distributions import Categorical

from src.environment.actions import Action


def select_action(actor, observation_vector, deterministic=False):
    observation = torch.tensor(
        observation_vector,
        dtype=torch.float32,
    ).unsqueeze(0)

    logits = actor(observation)

    distribution = Categorical(logits=logits)

    if deterministic:
        action_tensor = torch.argmax(logits, dim=-1)
    else:
        action_tensor = distribution.sample()

    action = Action(int(action_tensor.item()))
    log_prob = distribution.log_prob(action_tensor).item()

    return action, log_prob
