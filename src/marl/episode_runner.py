from src.communication.telemetry import TelemetryMessage
from src.marl.digital_twin_adapter import get_digital_twin_inputs


def run_episode(
    actor,
    critic,
    simulator,
    method,
    multi_agent_buffer,
    digital_twin=None,
    telemetry_channel=None,
    trusted_positions=None,
    reachable_occupancies=None,
    aoi_values=None,
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


        if telemetry_channel is not None:
            for agent_id, position in simulator.agent_positions.items():
                telemetry_channel.send(
                    TelemetryMessage(
                        agent_id=agent_id,
                        position=position,
                        source_timestamp=step,
                    ),
                    current_timestep=step,
                )

        if (
            telemetry_channel is not None
            and digital_twin is not None
        ):
            messages = telemetry_channel.receive_ready(
                current_timestep=step
            )

            for message in messages:
                digital_twin.process_telemetry(
                    message
                )




        if digital_twin is not None:
            dt_inputs = get_digital_twin_inputs(
                digital_twin=digital_twin,
                grid=simulator.grid,
                current_timestep=step,
            )

            trusted_positions = dt_inputs[
                "trusted_positions"
            ]

            reachable_occupancies = dt_inputs[
                "reachable_occupancies"
            ]

            aoi_values = dt_inputs[
                "aoi_values"
            ]

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


        if digital_twin is not None:
            for agent_id, action in enumerate(
                result["actions"]
            ):
                digital_twin.record_command(
                    agent_id=agent_id,
                    action=action,
                    timestep=step,
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
