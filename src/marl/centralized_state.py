import numpy as np


def build_centralized_state(agent_observations):
    """
    Concatenate all agent observation vectors into one global critic input.

    Parameters
    ----------
    agent_observations : list of np.ndarray
        One flattened observation vector per agent.

    Returns
    -------
    np.ndarray
        Concatenated centralized state.
    """
    if len(agent_observations) == 0:
        return np.array([], dtype=np.float32)

    return np.concatenate(
        [
            np.asarray(obs, dtype=np.float32)
            for obs in agent_observations
        ],
        axis=0,
    )
