def compute_reward(
    reached_goal=False,
    collision=False,
    deadlock=False,
    previous_distance=None,
    current_distance=None,
):
    reward = -0.02

    if reached_goal:
        reward += 10.0

    if collision:
        reward -= 10.0

    if deadlock:
        reward -= 5.0

    if (
        previous_distance is not None
        and current_distance is not None
    ):
        if current_distance < previous_distance:
            reward += 0.1
        elif current_distance > previous_distance:
            reward -= 0.1

    return reward
