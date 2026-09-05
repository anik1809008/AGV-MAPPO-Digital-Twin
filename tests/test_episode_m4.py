from src.digital_twin.digital_twin import DigitalTwin
from src.digital_twin.state import AgentTwinState
from src.environment.simulator import GroundTruthSimulator
from src.marl.episode_runner import run_episode
from src.marl.multi_agent_buffer import MultiAgentRolloutBuffer
from src.marl.networks import ActorNetwork, CriticNetwork
from src.safety.m4_controller import M4Controller
from src.safety.shield import SafetyShield


def test_episode_with_m4_controller():
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

    actor = ActorNetwork(
        input_dim=247,
        action_dim=5,
    )

    critic = CriticNetwork(
        input_dim=494,
    )

    shield = SafetyShield(grid)

    m4_controller = M4Controller(
        actor=actor,
        shield=shield,
    )

    buffer = MultiAgentRolloutBuffer(
        num_agents=2,
    )

    result = run_episode(
        actor=actor,
        critic=critic,
        simulator=simulator,
        method="M4",
        multi_agent_buffer=buffer,
        digital_twin=digital_twin,
        m4_controller=m4_controller,
        max_steps=4,
    )

    assert 1 <= result["steps"] <= 4
    assert len(buffer) == result["steps"]
    assert m4_controller.action_filter.intervention_count >= 0
