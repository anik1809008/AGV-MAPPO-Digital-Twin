import numpy as np


def build_local_observation(
    grid,
    center_position,
    trusted_positions,
    reachable_occupancies=None,
    window_size=9,
):
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")

    radius = window_size // 2

    # Channels:
    # 0 = obstacle / outside map
    # 1 = trusted AGV positions
    # 2 = reachable occupancy uncertainty
    observation = np.zeros(
        (3, window_size, window_size),
        dtype=np.float32,
    )

    center_x, center_y = center_position
    height = len(grid)
    width = len(grid[0])

    for local_y in range(window_size):
        for local_x in range(window_size):
            world_x = center_x + local_x - radius
            world_y = center_y + local_y - radius

            if (
                world_x < 0
                or world_x >= width
                or world_y < 0
                or world_y >= height
            ):
                observation[0, local_y, local_x] = 1.0
                continue

            if grid[world_y][world_x] == 1:
                observation[0, local_y, local_x] = 1.0

    for position in trusted_positions:
        dx = position[0] - center_x
        dy = position[1] - center_y

        local_x = dx + radius
        local_y = dy + radius

        if 0 <= local_x < window_size and 0 <= local_y < window_size:
            observation[1, local_y, local_x] = 1.0

    if reachable_occupancies is not None:
        for position in reachable_occupancies:
            dx = position[0] - center_x
            dy = position[1] - center_y

            local_x = dx + radius
            local_y = dy + radius

            if 0 <= local_x < window_size and 0 <= local_y < window_size:
                observation[2, local_y, local_x] = 1.0

    return observation
def build_m2_observation(
    grid,
    center_position,
    trusted_positions,
    window_size=9,
):
    return build_local_observation(
        grid=grid,
        center_position=center_position,
        trusted_positions=trusted_positions,
        reachable_occupancies=None,
        window_size=window_size,
    )


def build_m3_observation(
    grid,
    center_position,
    trusted_positions,
    reachable_occupancies,
    window_size=9,
):
    return build_local_observation(
        grid=grid,
        center_position=center_position,
        trusted_positions=trusted_positions,
        reachable_occupancies=reachable_occupancies,
        window_size=window_size,
    )
