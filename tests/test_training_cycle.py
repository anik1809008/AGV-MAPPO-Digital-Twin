from src.environment.simulator import GroundTruthSimulator
from src.marl.mappo_trainer import MAPPOTrainer
from src.marl.multi_agent_buffer import MultiAgentRolloutBuffer
from src.marl.networks import ActorNetwork, CriticNetwork
from src.marl.training_cycle import run_training_cycle


def test_run_training_cycle():
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

    trainer = MAPPOTrainer(
        actor=actor,
        critic=critic,
    )

    buffer = MultiAgentRolloutBuffer(
        num_agents=2,
    )

    result = run_training_cycle(
        actor=actor,
        critic=critic,
        trainer=trainer,
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
        max_steps=4,
        epochs=2,
        minibatch_size=4,
    )

    assert 1 <= result["episode"]["steps"] <= 4
    assert len(result["training_history"]) > 0
    assert len(buffer) == 0
