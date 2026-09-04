import numpy as np


def compute_gae(
    rewards,
    values,
    dones,
    next_value,
    gamma=0.99,
    gae_lambda=0.95,
):
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)

    advantages = np.zeros_like(rewards, dtype=np.float32)

    gae = 0.0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_non_terminal = 1.0 - dones[t]
            next_values = next_value
        else:
            next_non_terminal = 1.0 - dones[t]
            next_values = values[t + 1]

        delta = (
            rewards[t]
            + gamma * next_values * next_non_terminal
            - values[t]
        )

        gae = (
            delta
            + gamma
            * gae_lambda
            * next_non_terminal
            * gae
        )

        advantages[t] = gae

    returns = advantages + values

    return advantages, returns
