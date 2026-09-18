import argparse
import random

import numpy as np
import torch

from src.communication.latency_channel import FixedLatencyChannel
from src.digital_twin.digital_twin import DigitalTwin
from src.digital_twin.state import AgentTwinState
from src.environment.delayed_execution import DelayedCommandExecutor
from src.environment.execution_delay import ExecutionDelayModel
from src.environment.simulator import GroundTruthSimulator
from src.marl.checkpoint import save_checkpoint
from src.marl.checkpoint_paths import (
    build_training_checkpoint_path,
)
from src.marl.multi_agent_buffer import MultiAgentRolloutBuffer
from src.marl.scenario_split import get_random_scenario_path
from src.marl.training_components import build_mappo_components
from src.marl.training_cycle import run_training_cycle
from src.marl.training_instance import build_training_instance
from src.marl.training_schedule import build_training_schedule


MAP_PATH = (
    "benchmarks/movingai/"
    "warehouse-10-20-10-2-1.map"
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--method",
        choices=["M2", "M3", "M6"],
        required=True,
    )

    parser.add_argument(
        "--agents",
        type=int,
        choices=[8, 20],
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--episodes-per-scenario",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--latency",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--immediate-probability",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
    )

    args = parser.parse_args()

    if args.episodes_per_scenario < 1:
        raise ValueError(
            "episodes-per-scenario must be >= 1"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    components = build_mappo_components(
        agent_count=args.agents,
    )

    schedule = build_training_schedule()

    total_episodes = 0

    for scenario_id in schedule:
        scenario_path = get_random_scenario_path(
            scenario_id
        )

        for episode in range(
            args.episodes_per_scenario
        ):
            start_index = (
                episode * args.agents
            )

            instance = build_training_instance(
                map_path=MAP_PATH,
                scenario_path=scenario_path,
                agent_count=args.agents,
                start_index=start_index,
            )

            grid = instance["grid"]
            starts = instance["starts"]
            goals = instance["goals"]

            simulator = GroundTruthSimulator(
                grid=grid,
                agent_positions=starts,
                agent_goals=goals,
            )

            digital_twin = DigitalTwin({
                agent_id: AgentTwinState(
                    agent_id=agent_id,
                    last_trusted_position=(
                        starts[agent_id]
                    ),
                    last_trusted_timestamp=0,
                    goal=goals[agent_id],
                )
                for agent_id in starts
            })

            telemetry_channel = FixedLatencyChannel(
                latency_steps=args.latency,
            )

            execution_delay_model = (
                ExecutionDelayModel(
                    immediate_probability=(
                        args.immediate_probability
                    ),
                    seed=(
                        args.seed
                        + scenario_id * 1000
                        + episode
                    ),
                )
            )

            delayed_executor = DelayedCommandExecutor(
                delay_model=execution_delay_model,
            )

            buffer = MultiAgentRolloutBuffer(
                num_agents=args.agents,
            )

            result = run_training_cycle(
                actor=components["actor"],
                critic=components["critic"],
                trainer=components["trainer"],
                simulator=simulator,
                method=args.method,
                trusted_positions=starts,
                reachable_occupancies={
                    agent_id: {position}
                    for agent_id, position
                    in starts.items()
                },
                aoi_values={
                    agent_id: 0
                    for agent_id in starts
                },
                multi_agent_buffer=buffer,
                digital_twin=digital_twin,
                telemetry_channel=telemetry_channel,
                delayed_executor=delayed_executor,
                max_steps=args.max_steps,
            )

            total_episodes += 1

            print(
                f"Scenario {scenario_id}, "
                f"episode {episode + 1}: "
                f"steps={result['episode']['steps']}, "
                f"updates="
                f"{len(result['training_history'])}"
            )

    final_scenario_id = schedule[-1]

    checkpoint_path = (
        build_training_checkpoint_path(
            method=args.method,
            agent_count=args.agents,
            seed=args.seed,
            scenario_id=final_scenario_id,
        )
    )

    save_checkpoint(
        path=checkpoint_path,
        actor=components["actor"],
        critic=components["critic"],
        actor_optimizer=(
            components["trainer"].actor_optimizer
        ),
        critic_optimizer=(
            components["trainer"].critic_optimizer
        ),
        extra_state={
            "method": args.method,
            "agents": args.agents,
            "seed": args.seed,
            "latency": args.latency,
            "immediate_probability": (
                args.immediate_probability
            ),
            "episodes_per_scenario": (
                args.episodes_per_scenario
            ),
            "total_episodes": total_episodes,
            "training_scenario_ids": schedule,
            "final_training_scenario_id": (
                final_scenario_id
            ),
        },
    )

    print(
        "Final checkpoint saved:",
        checkpoint_path,
    )


if __name__ == "__main__":
    main()
