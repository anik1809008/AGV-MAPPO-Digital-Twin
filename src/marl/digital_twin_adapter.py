def get_digital_twin_inputs(
    digital_twin,
    grid,
    current_timestep,
):
    trusted_positions = {}
    reachable_occupancies = {}
    aoi_values = {}

    for agent_id, state in digital_twin.agent_states.items():
        trusted_positions[agent_id] = (
            state.last_trusted_position
        )

        aoi_values[agent_id] = (
            digital_twin.get_aoi(
                agent_id,
                current_timestep,
            )
        )

        reachable_occupancies[agent_id] = (
            digital_twin.get_reachable_occupancy(
                agent_id,
                grid,
            )
        )

    return {
        "trusted_positions": trusted_positions,
        "reachable_occupancies": reachable_occupancies,
        "aoi_values": aoi_values,
    }
