import argparse

from src.environment.simulator import GroundTruthSimulator
from src.mapf.lns_sat_runner import run_lns_sat_episode
from src.marl.scenario_split import get_random_scenario_path
from src.marl.training_instance import build_training_instance


MAP_PATH = (
    "benchmarks/movingai/"
    "warehouse-10-20-10-2-1.map"
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--agents",
        type=int,
        choices=[8, 20],
        required=True,
    )

    parser.add_argument(
        "--scenario-id",
        type=int,
        required=True,
    )

    args = parser.parse_args()

    scenario_path = get_random_scenario_path(
        args.scenario_id
    )

    instance = build_training_instance(
        map_path=MAP_PATH,
        scenario_path=scenario_path,
        agent_count=args.agents,
    )

    simulator = GroundTruthSimulator(
        grid=instance["grid"],
        agent_positions=instance["starts"],
        agent_goals=instance["goals"],
    )

    result = run_lns_sat_episode(
        simulator=simulator,
    )

    print(result)


if __name__ == "__main__":
    main()
