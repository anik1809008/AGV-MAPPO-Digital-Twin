"""Experimental M7 observations constructed without simulator positions."""

from src.marl.environment_adapter import build_all_agent_inputs


def build_dt_only_agent_inputs(
    *,
    grid,
    agent_goals,
    trusted_positions,
    reachable_occupancies,
    aoi_values,
    observation_centers=None,
):
    agent_ids = set(agent_goals)

    if agent_ids != set(range(len(agent_goals))):
        raise ValueError("Agent IDs must be consecutive from zero")

    if not (
        set(trusted_positions) == agent_ids
        and set(reachable_occupancies) == agent_ids
        and set(aoi_values) == agent_ids
    ):
        raise ValueError("Complete Digital Twin observations are required")

    if observation_centers is None:
        observation_centers = trusted_positions

    if set(observation_centers) != agent_ids:
        raise ValueError("Complete observation centers required")

    centers = [
        tuple(observation_centers[agent_id])
        for agent_id in sorted(agent_ids)
    ]

    goals = [
        tuple(agent_goals[agent_id])
        for agent_id in sorted(agent_ids)
    ]

    return build_all_agent_inputs(
        method="M7",
        grid=grid,
        agent_positions=centers,
        agent_goals=goals,
        trusted_positions=trusted_positions,
        reachable_occupancies=reachable_occupancies,
        aoi_values=aoi_values,
    )
