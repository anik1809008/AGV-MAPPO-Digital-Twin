import numpy as np
import torch

from src.marl.centralized_state import build_centralized_state
from src.marl.policy import select_action


def collect_single_step(
    actor,
    critic,
    agent_observations,
    deterministic=False,
):
    """
    Collect one MAPPO decision step for all agents.

    Parameters
    ----------
    actor
        Shared actor network.

    critic
        Centralized critic network.

    agent_observations
        List of flattened local observation vectors,
        one per agent.

    deterministic
        If True, use argmax actions.
        Otherwise sample from the policy.

    Returns
    -------
    dict
        Actions, log probabilities, value estimate,
        and centralized state.
    """

    centralized_state = build_centralized_state(
        agent_observations
    )

    critic_input = torch.tensor(
        centralized_state,
        dtype=torch.float32,
    ).unsqueeze(0)

    with torch.no_grad():
        value = critic(
            critic_input
        ).squeeze().item()

    actions = []
    log_probs = []

    for observation in agent_observations:
        action, log_prob = select_action(
            actor=actor,
            observation_vector=np.asarray(
                observation,
                dtype=np.float32,
            ),
            deterministic=deterministic,
        )

        actions.append(action)
        log_probs.append(float(log_prob))

    return {
        "actions": actions,
        "log_probs": log_probs,
        "value": float(value),
        "centralized_state": centralized_state,
    }
