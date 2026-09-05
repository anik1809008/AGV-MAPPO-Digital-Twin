from src.marl.reward import compute_reward


def execute_training_step(
    simulator,
    actions,
):
    previous_positions = list(
        simulator.agent_positions.values()
    )

    previous_distances = [
        simulator.manhattan_distance(
            previous_positions[agent_id],
            simulator.agent_goals[agent_id],
        )
        for agent_id in range(
            len(previous_positions)
        )
    ]

    result = simulator.step_joint(
        actions
    )

    current_positions = list(
        simulator.agent_positions.values()
    )

    rewards = []

    collision = bool(
        result.get("collision", False)
    )

    for agent_id, position in enumerate(
        current_positions
    ):
        current_distance = (
            simulator.manhattan_distance(
                position,
                simulator.agent_goals[
                    agent_id
                ],
            )
        )

        reached_goal = simulator.is_at_goal(
            agent_id
        )

        deadlock = simulator.is_deadlocked()

        reward = compute_reward(
            previous_distance=previous_distances[
                agent_id
            ],
            current_distance=current_distance,
            reached_goal=reached_goal,
            collision=collision,
            deadlock=deadlock,
        )

        rewards.append(reward)

    done = (
        collision
        or simulator.all_goals_reached()
    )

    dones = [
        done
        for _ in current_positions
    ]

    return {
        "previous_positions": previous_positions,
        "current_positions": current_positions,
        "rewards": rewards,
        "dones": dones,
        "collision": collision,
        "all_goals_reached": simulator.all_goals_reached(),
    }
