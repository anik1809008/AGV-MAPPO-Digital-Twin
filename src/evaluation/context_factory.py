from src.communication.latency_channel import FixedLatencyChannel
from src.digital_twin.digital_twin import DigitalTwin
from src.digital_twin.state import AgentTwinState
from src.environment.delayed_execution import DelayedCommandExecutor
from src.environment.execution_delay import ExecutionDelayModel
from src.environment.simulator import GroundTruthSimulator
from src.marl.multi_agent_buffer import MultiAgentRolloutBuffer


def build_experiment_context(
    grid,
    starts,
    goals,
    actor,
    critic,
    latency_steps,
    max_steps,
    m4_controller=None,
    m5_baseline=None,
    execution_delay_model=None,
):
    simulator = GroundTruthSimulator(
        grid=grid,
        agent_positions=dict(starts),
        agent_goals=dict(goals),
    )

    digital_twin = DigitalTwin({
        agent_id: AgentTwinState(
            agent_id=agent_id,
            last_trusted_position=starts[agent_id],
            last_trusted_timestamp=0,
            goal=goals[agent_id],
        )
        for agent_id in starts
    })

    telemetry_channel = FixedLatencyChannel(
        latency_steps=latency_steps,
    )

    if execution_delay_model is None:
        execution_delay_model = ExecutionDelayModel(
            immediate_probability=0.8,
        )

    delayed_executor = DelayedCommandExecutor(
        delay_model=execution_delay_model,
    )

    buffer = MultiAgentRolloutBuffer(
        num_agents=len(starts),
    )

    return {
        "simulator": simulator,
        "digital_twin": digital_twin,
        "telemetry_channel": telemetry_channel,
        "delayed_executor": delayed_executor,
        "buffer": buffer,
        "actor": actor,
        "critic": critic,
        "m4_controller": m4_controller,
        "m5_baseline": m5_baseline,
        "max_steps": max_steps,
    }
