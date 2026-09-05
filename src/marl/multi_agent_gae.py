import numpy as np

from src.marl.gae import compute_gae


def compute_multi_agent_gae(
    multi_agent_buffer,
    next_values=None,
    gamma=0.99,
    gae_lambda=0.95,
):
    num_agents = multi_agent_buffer.num_agents

    if next_values is None:
        next_values = [0.0] * num_agents

    all_advantages = []
    all_returns = []

    for agent_id in range(num_agents):
        buffer = multi_agent_buffer.agent_buffers[agent_id]

        advantages, returns = compute_gae(
            rewards=buffer.rewards,
            values=buffer.values,
            dones=buffer.dones,
            next_value=next_values[agent_id],
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        all_advantages.append(advantages)
        all_returns.append(returns)

    return (
        np.asarray(all_advantages, dtype=np.float32),
        np.asarray(all_returns, dtype=np.float32),
    )
