from src.environment.movingai_scenario import load_movingai_scenario
import argparse
from src.environment.delayed_execution import DelayedCommandExecutor
from src.environment.execution_delay import ExecutionDelayModel
from src.communication.latency_channel import FixedLatencyChannel
from src.digital_twin.digital_twin import DigitalTwin
from src.digital_twin.state import AgentTwinState
from src.environment.simulator import GroundTruthSimulator
from src.marl.multi_agent_buffer import MultiAgentRolloutBuffer
from src.marl.training_components import build_mappo_components
from src.marl.training_cycle import run_training_cycle
from src.marl.training_instance import build_training_instance


from src.marl.checkpoint import (
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint_compatibility,
)





import random

import numpy as np
import torch




def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--method",
        choices=["M2", "M3", "M6"],
        default="M3",
    )
    parser.add_argument(
        "--agents",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--latency",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="results/mappo_checkpoint.pt",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )


    parser.add_argument(
        "--immediate-probability",
        type=float,
        default=0.8,
    )


    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )




    args = parser.parse_args()


    scenario_path = (
        "benchmarks/movingai/scen-random/"
        "warehouse-10-20-10-2-1-random-1.scen"
    )

    scenarios = load_movingai_scenario(
        scenario_path
    )











    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    components = build_mappo_components(
        agent_count=args.agents,
    )

    previous_episodes = 0
    effective_start_index = args.start_index




    if args.resume:
        extra_state = load_checkpoint(
            path=args.checkpoint,
            actor=components["actor"],
            critic=components["critic"],
            actor_optimizer=(
                components["trainer"].actor_optimizer
            ),
            critic_optimizer=(
                components["trainer"].critic_optimizer
            ),
        )




        validate_checkpoint_compatibility(
            extra_state=extra_state,
            method=args.method,
            agents=args.agents,
            latency=args.latency,
            immediate_probability=(
                args.immediate_probability
            ),
        )





        previous_episodes = extra_state.get(
            "total_episodes",
            extra_state.get("episodes", 0),
        )

        checkpoint_start_index = extra_state.get(
            "start_index",
            0,
        )

        checkpoint_episodes = extra_state.get(
            "episodes",
            0,
        )

        effective_start_index = (
            checkpoint_start_index
            + checkpoint_episodes * args.agents
        )

        print(
            "Checkpoint loaded:",
            args.checkpoint,
        )

        print(
            "Checkpoint state:",
            extra_state,
        )

    required_scenarios = (
        effective_start_index
        + args.episodes * args.agents
    )

    if required_scenarios > len(scenarios):
        raise ValueError(
            "Requested training range exceeds "
            f"available scenarios: need "
            f"{required_scenarios}, "
            f"available {len(scenarios)}"
        )

    for episode in range(args.episodes):



        start_index = (
            effective_start_index
            + episode * args.agents
        )



        instance = build_training_instance(
            map_path=(
                "benchmarks/movingai/"
                "warehouse-10-20-10-2-1.map"
            ),
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
                last_trusted_position=starts[agent_id],
                last_trusted_timestamp=0,
                goal=goals[agent_id],
            )
            for agent_id in starts
        })

        telemetry_channel = FixedLatencyChannel(
            latency_steps=args.latency,
        )


        execution_delay_model = ExecutionDelayModel(
            immediate_probability=args.immediate_probability,
            seed=args.seed + episode,
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
            delayed_executor=delayed_executor,
            method=args.method,
            trusted_positions=starts,
            reachable_occupancies={
                agent_id: {position}
                for agent_id, position in starts.items()
            },
            aoi_values={
                agent_id: 0
                for agent_id in starts
            },
            multi_agent_buffer=buffer,
            digital_twin=digital_twin,
            telemetry_channel=telemetry_channel,
            max_steps=args.max_steps,
        )

        print(
            f"Episode {episode + 1}: "
            f"steps={result['episode']['steps']}, "
            f"updates={len(result['training_history'])}"
        )



    total_episodes = previous_episodes + args.episodes

    save_checkpoint(
        path=args.checkpoint,
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
            "episodes": args.episodes,
            "total_episodes": total_episodes,
            "latency": args.latency,
            "start_index": effective_start_index,
            "immediate_probability": args.immediate_probability,
            "seed": args.seed,
        },
    )



    print(
        "Checkpoint saved:",
        args.checkpoint,
    )


if __name__ == "__main__":
    main()
