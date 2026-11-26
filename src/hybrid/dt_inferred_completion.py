"""Eligibility checks for experimental DT-inferred completion."""


def get_inferred_completion_positions(
    *,
    dt,
    goals,
    immediate_probability,
    pending_commands,
    handover_active=False,
    max_aoi=2,
):
    """Return eligible DT-inferred positions or None."""

    if max_aoi < 1:
        raise ValueError("max_aoi must be >= 1")

    if immediate_probability != 1.0:
        return None

    if pending_commands or handover_active:
        return None

    candidates = dt.get("possible_current_positions", {})
    ages = dt.get("aoi_values", {})
    agent_ids = set(goals)

    if not agent_ids:
        return None

    if set(candidates) != agent_ids:
        return None

    if set(ages) != agent_ids:
        return None

    if any(not 0 <= age <= max_aoi for age in ages.values()):
        return None

    # Preserve the existing fresh-state recovery path.
    if not any(age > 0 for age in ages.values()):
        return None

    positions = {}

    for agent_id in sorted(agent_ids):
        possible = candidates[agent_id]

        if len(possible) != 1:
            return None

        positions[agent_id] = tuple(next(iter(possible)))

    # Reject inconsistent inferred occupancy.
    if len(set(positions.values())) != len(positions):
        return None

    unfinished = [
        agent_id
        for agent_id in agent_ids
        if positions[agent_id] != tuple(goals[agent_id])
    ]

    if len(unfinished) not in (1, 2):
        return None

    return positions
