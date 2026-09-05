from src.environment.simulator import GroundTruthSimulator
from src.marl.episode_runner import run_episode
from src.marl.multi_agent_buffer import MultiAgentRolloutBuffer
from src.marl.networks import ActorNetwork, CriticNetwork


def test_run_episode_respects_max_steps():
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

    result = run_episode(
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
        max_steps=5,
    )

    assert 1 <= result["steps"] <= 5
    assert len(buffer) == result["steps"]
    assert len(result["total_rewards"]) == 2
    assert isinstance(result["collision"], bool)
    assert isinstance(result["all_goals_reached"], bool)
