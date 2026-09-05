from src.communication.latency_channel import FixedLatencyChannel
from src.digital_twin.digital_twin import DigitalTwin
from src.digital_twin.state import AgentTwinState
from src.environment.delayed_execution import DelayedCommandExecutor
from src.environment.execution_delay import ExecutionDelayModel
from src.environment.simulator import GroundTruthSimulator
from src.marl.episode_runner import run_episode
from src.marl.multi_agent_buffer import MultiAgentRolloutBuffer
from src.marl.networks import ActorNetwork, CriticNetwork


def test_episode_with_telemetry_and_execution_delay():
    grid = [
        [0, 0, 0],
        [0, 0, 0],
    ]

    simulator = GroundTruthSimulator(
        grid=grid,
        agent_positions={
            0: (0, 0),
            1: (2, 1),
        },
        agent_goals={
            0: (2, 0),
            1: (0, 1),
        },
    )

    digital_twin = DigitalTwin({
        0: AgentTwinState(
            agent_id=0,
            last_trusted_position=(0, 0),
            last_trusted_timestamp=0,
            goal=(2, 0),
        ),
        1: AgentTwinState(
            agent_id=1,
            last_trusted_position=(2, 1),
            last_trusted_timestamp=0,
            goal=(0, 1),
        ),
    })

    telemetry_channel = FixedLatencyChannel(
        latency_steps=2
    )

    delayed_executor = DelayedCommandExecutor(
        ExecutionDelayModel(
            immediate_probability=0.0,
            seed=1,
        )
    )

    actor = ActorNetwork(
        input_dim=247,
        action_dim=5,
    )

    critic = CriticNetwork(
        input_dim=494,
    )

    buffer = MultiAgentRolloutBuffer(
        num_agents=2,
    )

    result = run_episode(
        actor=actor,
        critic=critic,
        simulator=simulator,
        method="M3",
        multi_agent_buffer=buffer,
        digital_twin=digital_twin,
        telemetry_channel=telemetry_channel,
        delayed_executor=delayed_executor,
        max_steps=4,
    )

    assert 1 <= result["steps"] <= 4
    assert len(buffer) == result["steps"]

    for agent_id in digital_twin.agent_states:
        assert digital_twin.get_aoi(
            agent_id,
            result["steps"] - 1,
        ) >= 0

    for queue in delayed_executor.pending_commands.values():
        assert len(queue) >= 0
