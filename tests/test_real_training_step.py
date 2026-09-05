from src.environment.simulator import GroundTruthSimulator
from src.marl.multi_agent_buffer import MultiAgentRolloutBuffer
from src.marl.networks import ActorNetwork, CriticNetwork
from src.marl.real_training_step import run_real_training_step


def test_run_real_training_step_stores_buffer_data():
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

    result = run_real_training_step(
        actor=actor,
        critic=critic,
        simulator=simulator,
        method="M3",
        trusted_positions={
            0: (0, 0),
            1: (2, 1),
        },
        reachable_occupancies={
            0: {(0, 0)},
            1: {(2, 1)},
        },
        aoi_values={
            0: 0,
            1: 0,
        },
        multi_agent_buffer=buffer,
    )

    assert len(buffer) == 1
    assert len(buffer.agent_buffers[0]) == 1
    assert len(buffer.agent_buffers[1]) == 1
    assert len(result["actions"]) == 2
    assert len(result["rewards"]) == 2
