"""Select observation centers from controller-visible DT estimates."""


def select_dt_observation_centers(
    trusted_positions,
    possible_current_positions,
):
    centers = {}

    for agent_id, trusted in trusted_positions.items():
        candidates = set(
            possible_current_positions.get(agent_id, ())
        )

        if not candidates:
            raise ValueError(
                f"Missing possible positions for AGV {agent_id}"
            )

        if len(candidates) == 1:
            centers[agent_id] = next(iter(candidates))
        else:
            centers[agent_id] = tuple(trusted)

    return centers
