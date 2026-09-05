import numpy as np


def flatten_multi_agent_rollout(
    multi_agent_buffer,
):
    observations = []
    actions = []
    log_probs = []
    rewards = []
    values = []
    dones = []
    centralized_states = []

    num_timesteps = len(
        multi_agent_buffer.centralized_states
    )

    for timestep in range(num_timesteps):
        centralized_state = (
            multi_agent_buffer.centralized_states[
                timestep
            ]
        )

        for agent_buffer in (
            multi_agent_buffer.agent_buffers
        ):
            observations.append(
                agent_buffer.observations[timestep]
            )

            actions.append(
                agent_buffer.actions[timestep]
            )

            log_probs.append(
                agent_buffer.log_probs[timestep]
            )

            rewards.append(
                agent_buffer.rewards[timestep]
            )

            values.append(
                agent_buffer.values[timestep]
            )

            dones.append(
                agent_buffer.dones[timestep]
            )

            centralized_states.append(
                centralized_state
            )

    return {
        "observations": np.asarray(
            observations,
            dtype=np.float32,
        ),
        "actions": np.asarray(
            actions,
            dtype=np.int64,
        ),
        "log_probs": np.asarray(
            log_probs,
            dtype=np.float32,
        ),
        "rewards": np.asarray(
            rewards,
            dtype=np.float32,
        ),
        "values": np.asarray(
            values,
            dtype=np.float32,
        ),
        "dones": np.asarray(
            dones,
            dtype=np.bool_,
        ),
        "centralized_states": np.asarray(
            centralized_states,
            dtype=np.float32,
        ),
    }
