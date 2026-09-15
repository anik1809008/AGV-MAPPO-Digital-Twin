import argparse

from src.communication.latency_channel import FixedLatencyChannel
from src.digital_twin.digital_twin import DigitalTwin
from src.digital_twin.state import AgentTwinState
from src.environment.simulator import GroundTruthSimulator
from src.marl.multi_agent_buffer import MultiAgentRolloutBuffer
from src.marl.training_components import build_mappo_components
from src.marl.training_cycle import run_training_cycle
from src.marl.training_instance import build_training_instance
from src.marl.checkpoint import load_checkpoint, save_checkpoint

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




    args = parser.parse_args()

    components = build_mappo_components(
        agent_count=args.agents,
    )

    previous_episodes = 0


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

        previous_episodes = extra_state.get(
            "total_episodes",
            extra_state.get("episodes", 0),
        )

        print(
            "Checkpoint loaded:",
            args.checkpoint,
        )
        print(
            "Checkpoint state:",
            extra_state,
        )



    for episode in range(args.episodes):
        start_index = (
            args.start_index
            + episode * args.agents
        )

        instance = build_training_instance(
            map_path=(
                "benchmarks/movingai/"
                "warehouse-10-20-10-2-1.map"
            ),
            scenario_path=(
                "benchmarks/movingai/scen-random/"
                "warehouse-10-20-10-2-1-random-1.scen"
            ),
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
            "start_index": args.start_index,
            
        },
    )

    total_episodes = previous_episodes + args.episodes


    print(
        "Checkpoint saved:",
        args.checkpoint,
    )


if __name__ == "__main__":
    main()
