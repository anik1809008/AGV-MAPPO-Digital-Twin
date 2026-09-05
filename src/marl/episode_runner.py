def run_episode(
    actor,
    critic,
    simulator,
    method,
    trusted_positions,
    reachable_occupancies,
    aoi_values,
    multi_agent_buffer,
    max_steps=200,
):
    from src.marl.real_training_step import run_real_training_step

    total_rewards = [
        0.0
        for _ in simulator.agent_positions
    ]

    collision = False
    all_goals_reached = False
    steps = 0

    for step in range(max_steps):
        result = run_real_training_step(
            actor=actor,
            critic=critic,
            simulator=simulator,
            method=method,
            trusted_positions=trusted_positions,
            reachable_occupancies=reachable_occupancies,
            aoi_values=aoi_values,
            multi_agent_buffer=multi_agent_buffer,
        )

        for agent_id, reward in enumerate(
            result["rewards"]
        ):
            total_rewards[agent_id] += reward

        steps = step + 1
        collision = result["collision"]

        all_goals_reached = (
            simulator.all_goals_reached()
        )

        if collision or all_goals_reached:
            break

    return {
        "steps": steps,
        "total_rewards": total_rewards,
        "collision": collision,
        "all_goals_reached": all_goals_reached,
    }
