from src.marl.episode_runner import run_episode


def run_marl_method(
    method,
    actor,
    critic,
    simulator,
    multi_agent_buffer,
    digital_twin=None,
    telemetry_channel=None,
    delayed_executor=None,
    m4_controller=None,
    m5_baseline=None,
    trusted_positions=None,
    reachable_occupancies=None,
    aoi_values=None,
    max_steps=200,
):
    if method not in {
        "M2",
        "M3",
        "M4",
        "M5",
    }:
        raise ValueError(
            "method must be one of: M2, M3, M4, M5"
        )

    return run_episode(
        actor=actor,
        critic=critic,
        simulator=simulator,
        method=method,
        multi_agent_buffer=multi_agent_buffer,
        digital_twin=digital_twin,
        telemetry_channel=telemetry_channel,
        delayed_executor=delayed_executor,
        m4_controller=m4_controller,
        m5_baseline=m5_baseline,
        trusted_positions=trusted_positions,
        reachable_occupancies=reachable_occupancies,
        aoi_values=aoi_values,
        max_steps=max_steps,
    )
