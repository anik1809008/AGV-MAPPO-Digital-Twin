"""Controller-visible Digital Twin decision support for experimental M7."""


def get_dt_decision_positions(trusted_positions, possible_positions, agent_ids):
    """Use trusted telemetry, never simulator ground truth."""
    result = {}

    for agent_id in agent_ids:
        if agent_id not in trusted_positions:
            raise ValueError(f"Missing trusted position: {agent_id}")

        candidates = set(possible_positions.get(agent_id, ()))

        if not candidates:
            raise ValueError(f"Missing possible positions: {agent_id}")

        trusted = tuple(trusted_positions[agent_id])
        result[agent_id] = trusted

    return result


def update_dt_progress(monitor, agent_id, position, goal, aoi):
    """Update observed progress only from fresh telemetry."""
    if aoi != 0:
        return monitor.is_stagnating(agent_id)

    return monitor.update(
        agent_id=agent_id,
        position=position,
        goal=goal,
    )


def select_dt_recovery_action(
    planner, agent_id, possible_positions, reachable_occupancies, goal
):
    """Recover only when the agent position is unambiguous."""
    candidates = set(possible_positions[agent_id])

    if len(candidates) != 1:
        return None

    occupied = set()
    for other_id, positions in reachable_occupancies.items():
        if other_id != agent_id:
            occupied.update(positions)

    return planner.select_recovery_action(
        agent_id=agent_id,
        current_position=next(iter(candidates)),
        goal_position=goal,
        occupied_positions=occupied,
    )


def reserve_dt_next_positions(shield, possible_positions, action):
    """Reserve all plausible occupied positions after an action."""
    if not possible_positions:
        raise ValueError("Possible positions cannot be empty")

    return shield.possible_next_positions(
        possible_current_positions=set(possible_positions),
        action=action,
    )
