def compute_step_path_length(
    previous_positions,
    current_positions,
):
    total = 0

    for agent_id in previous_positions:
        if (
            previous_positions[agent_id]
            != current_positions[agent_id]
        ):
            total += 1

    return total
