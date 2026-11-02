"""Conservative criteria for attempting command-drain recovery."""


def should_start_recovery(
    *,
    mode,
    timestep,
    last_switch_timestep,
    aoi_values,
    min_dwell_steps=3,
):
    if min_dwell_steps < 1:
        raise ValueError("min_dwell_steps must be >= 1")

    return (
        mode == "M7"
        and last_switch_timestep is not None
        and timestep - last_switch_timestep >= min_dwell_steps
        and bool(aoi_values)
        and all(age == 0 for age in aoi_values.values())
    )
