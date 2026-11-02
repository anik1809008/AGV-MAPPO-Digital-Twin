"""Verify post-drain state before allowing classical replanning."""


def recovery_state_confirmed(
    *,
    digital_twin,
    aoi_values,
    possible_current_positions,
    pending_commands,
    last_drain_timestep,
):
    if last_drain_timestep is None:
        return False

    agent_ids = set(digital_twin.agent_states)

    if not agent_ids:
        return False

    if set(aoi_values) != agent_ids:
        return False

    if set(possible_current_positions) != agent_ids:
        return False

    if pending_commands:
        return False

    for agent_id in agent_ids:
        state = digital_twin.agent_states[agent_id]

        if state.last_trusted_timestamp <= last_drain_timestep:
            return False

        if aoi_values[agent_id] != 0:
            return False

        if len(possible_current_positions[agent_id]) != 1:
            return False

    return True
