from src.marl.observation_builder import build_agent_input


def build_all_agent_inputs(
    method,
    grid,
    agent_positions,
    agent_goals,
    trusted_positions,
    reachable_occupancies,
    aoi_values,
):
    observations = []

    for agent_id, position in enumerate(agent_positions):
        observation = build_agent_input(
            method=method,
            grid=grid,
            center_position=position,
            goal_position=agent_goals[agent_id],
            trusted_positions=list(trusted_positions.values()),
            reachable_occupancies=set().union(
                *reachable_occupancies.values()
            ),


            aoi=aoi_values[agent_id],
            reachable_size=len(
                reachable_occupancies.get(
                    agent_id,
                    {position},
                )
            ),
            window_size=9,
        )

        observations.append(observation)

    return observations
