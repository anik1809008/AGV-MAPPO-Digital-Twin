from src.environment.delayed_execution import DelayedCommandExecutor
from src.environment.execution_delay import ExecutionDelayModel

from src.environment.simulator import GroundTruthSimulator
from src.marl.mappo_trainer import MAPPOTrainer
from src.marl.multi_agent_buffer import MultiAgentRolloutBuffer
from src.marl.networks import ActorNetwork, CriticNetwork
from src.marl.training_cycle import run_training_cycle
from src.communication.latency_channel import FixedLatencyChannel
from src.digital_twin.digital_twin import DigitalTwin
from src.digital_twin.state import AgentTwinState


import random

import numpy as np
import torch





def test_training_cycle_with_dynamic_digital_twin():
    grid = [
        [0, 0, 0],
        [0, 0, 0],
    ]

    starts = {
        0: (0, 0),
        1: (2, 1),
    }

    goals = {
        0: (2, 0),
        1: (0, 1),
    }

    simulator = GroundTruthSimulator(
        grid=grid,
        agent_positions=starts,
        agent_goals=goals,
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
        latency_steps=2,
    )

    actor = ActorNetwork(
        input_dim=249,
        action_dim=5,
    )

    critic = CriticNetwork(
        input_dim=498,
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
        trusted_positions=starts,
        reachable_occupancies={
            0: {(0, 0)},
            1: {(2, 1)},
        },
        aoi_values={
            0: 0,
            1: 0,
        },
        multi_agent_buffer=buffer,
        digital_twin=digital_twin,
        telemetry_channel=telemetry_channel,
        max_steps=4,
        epochs=2,
        minibatch_size=4,
    )

    assert 1 <= result["episode"]["steps"] <= 4
    assert len(result["training_history"]) > 0
    assert len(buffer) == 0






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
        input_dim=249,
        action_dim=5,
    )

    critic = CriticNetwork(
        input_dim=498,
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
def test_m6_training_cycle_with_dynamic_digital_twin():
    grid = [
        [0, 0, 0],
        [0, 0, 0],
    ]

    starts = {
        0: (0, 0),
        1: (2, 1),
    }

    goals = {
        0: (2, 0),
        1: (0, 1),
    }

    simulator = GroundTruthSimulator(
        grid=grid,
        agent_positions=starts,
        agent_goals=goals,
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
        latency_steps=2,
    )

    actor = ActorNetwork(
        input_dim=249,
        action_dim=5,
    )

    critic = CriticNetwork(
        input_dim=498,
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
        method="M6",
        trusted_positions=starts,
        reachable_occupancies={
            0: {(0, 0)},
            1: {(2, 1)},
        },
        aoi_values={
            0: 0,
            1: 0,
        },
        multi_agent_buffer=buffer,
        digital_twin=digital_twin,
        telemetry_channel=telemetry_channel,
        max_steps=4,
        epochs=2,
        minibatch_size=4,
    )

    assert 1 <= result["episode"]["steps"] <= 4
    assert len(result["training_history"]) > 0
    assert len(buffer) == 0
def test_seeded_training_is_reproducible():
    def run_once(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        grid = [
            [0, 0, 0],
            [0, 0, 0],
        ]

        starts = {
            0: (0, 0),
            1: (2, 1),
        }

        goals = {
            0: (2, 0),
            1: (0, 1),
        }

        simulator = GroundTruthSimulator(
            grid=grid,
            agent_positions=starts,
            agent_goals=goals,
        )

        actor = ActorNetwork(
            input_dim=249,
            action_dim=5,
        )

        critic = CriticNetwork(
            input_dim=498,
        )

        trainer = MAPPOTrainer(
            actor=actor,
            critic=critic,
        )

        buffer = MultiAgentRolloutBuffer(
            num_agents=2,
        )

        run_training_cycle(
            actor=actor,
            critic=critic,
            trainer=trainer,
            simulator=simulator,
            method="M6",
            trusted_positions=starts,
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

        actor_state = {
            key: value.detach().clone()
            for key, value in actor.state_dict().items()
        }

        critic_state = {
            key: value.detach().clone()
            for key, value in critic.state_dict().items()
        }

        return actor_state, critic_state

    actor_a, critic_a = run_once(seed=42)
    actor_b, critic_b = run_once(seed=42)

    for key in actor_a:
        assert torch.equal(
            actor_a[key],
            actor_b[key],
        )

    for key in critic_a:
        assert torch.equal(
            critic_a[key],
            critic_b[key],
        )
def test_training_cycle_with_execution_delay():
    grid = [
        [0, 0, 0],
        [0, 0, 0],
    ]

    starts = {
        0: (0, 0),
        1: (2, 1),
    }

    goals = {
        0: (2, 0),
        1: (0, 1),
    }

    simulator = GroundTruthSimulator(
        grid=grid,
        agent_positions=starts,
        agent_goals=goals,
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
        latency_steps=2,
    )

    delayed_executor = DelayedCommandExecutor(
        ExecutionDelayModel(
            immediate_probability=0.0,
            seed=1,
        )
    )

    actor = ActorNetwork(
        input_dim=249,
        action_dim=5,
    )

    critic = CriticNetwork(
        input_dim=498,
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
        method="M6",
        trusted_positions=starts,
        reachable_occupancies={
            0: {(0, 0)},
            1: {(2, 1)},
        },
        aoi_values={
            0: 0,
            1: 0,
        },
        multi_agent_buffer=buffer,
        digital_twin=digital_twin,
        telemetry_channel=telemetry_channel,
        delayed_executor=delayed_executor,
        max_steps=4,
        epochs=2,
        minibatch_size=4,
    )

    assert 1 <= result["episode"]["steps"] <= 4
    assert len(result["training_history"]) > 0
    assert len(buffer) == 0
