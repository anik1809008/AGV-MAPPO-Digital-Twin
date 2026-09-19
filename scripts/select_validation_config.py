import argparse

from src.evaluation.validation_selection import (
    group_validation_candidates,
    load_validation_rows,
    select_best_validation_candidate,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results-path",
        type=str,
        default="results/validation_sweep.csv",
    )

    parser.add_argument(
        "--method",
        choices=[
            "M2",
            "M3",
            "M4",
            "M5",
            "M6",
        ],
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

    args = parser.parse_args()

    rows = load_validation_rows(
        args.results_path
    )

    candidate_rows = (
        group_validation_candidates(
            rows=rows,
            method=args.method,
            agent_count=args.agents,
            seed=args.seed,
        )
    )

    if not candidate_rows:
        raise ValueError(
            "No matching validation results found"
        )

    result = select_best_validation_candidate(
        candidate_rows
    )

    candidate = result["candidate"]
    summary = result["summary"]

    print(
        "Selected validation configuration:"
    )

    print(
        f"method={args.method}"
    )
    print(
        f"agents={args.agents}"
    )
    print(
        f"seed={args.seed}"
    )

    if args.method == "M5":
        budget, threshold = candidate

        print(
            f"episodes_per_scenario={budget}"
        )
        print(
            f"m5_threshold={threshold}"
        )
    else:
        print(
            f"episodes_per_scenario={candidate}"
        )

    print(
        "Validation summary:"
    )

    print(
        f"collision_rate="
        f"{summary['collision_rate']}"
    )
    print(
        f"deadlock_rate="
        f"{summary['deadlock_rate']}"
    )
    print(
        f"success_rate="
        f"{summary['success_rate']}"
    )
    print(
        f"mean_makespan="
        f"{summary['mean_makespan']}"
    )
    print(
        f"mean_path_length="
        f"{summary['mean_path_length']}"
    )
    print(
        f"mean_planning_time="
        f"{summary['mean_planning_time']}"
    )


if __name__ == "__main__":
    main()
