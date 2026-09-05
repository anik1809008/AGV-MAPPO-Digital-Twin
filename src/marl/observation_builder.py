from src.marl.observations import (
    build_m2_observation,
    build_m3_observation,
    build_scalar_features,
    flatten_mappo_input,
)


def build_agent_input(
    method,
    grid,
    center_position,
    goal_position,
    trusted_positions,
    reachable_occupancies,
    aoi,
    reachable_size,
    window_size=9,
):
    if method == "M2":
        spatial = build_m2_observation(
            grid=grid,
            center_position=center_position,
            trusted_positions=trusted_positions,
            window_size=window_size,
        )

        scalars = build_scalar_features(
            center_position=center_position,
            goal_position=goal_position,
            aoi=0,
            reachable_size=1,
        )

    elif method in {"M3", "M4", "M5"}:
        spatial = build_m3_observation(
            grid=grid,
            center_position=center_position,
            trusted_positions=trusted_positions,
            reachable_occupancies=reachable_occupancies,
            window_size=window_size,
        )

        scalars = build_scalar_features(
            center_position=center_position,
            goal_position=goal_position,
            aoi=aoi,
            reachable_size=reachable_size,
        )

    else:
        raise ValueError(
            "method must be one of: M2, M3, M4,M5"
        )

    return flatten_mappo_input(
        spatial,
        scalars,
    )
