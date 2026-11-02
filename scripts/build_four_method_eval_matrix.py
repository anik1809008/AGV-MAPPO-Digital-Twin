"""Create a reproducible 250-configuration evaluation matrix."""

import argparse
import csv
from itertools import product
from pathlib import Path

MAP_PATH = "benchmarks/movingai/warehouse-10-20-10-2-1.map"
SCENARIOS = range(16, 21)
SEEDS = range(5)
LATENCIES = range(5)
PROBABILITIES = (1.0, 0.8)

FIELDS = [
    "episode_id",
    "map_path",
    "scenario_id",
    "agent_count",
    "start_index",
    "seed",
    "latency_steps",
    "immediate_probability",
]


def build_matrix():
    rows = []

    for scenario, seed, latency, probability in product(
        SCENARIOS, SEEDS, LATENCIES, PROBABILITIES
    ):
        rows.append({
            "episode_id": len(rows) + 1,
            "map_path": MAP_PATH,
            "scenario_id": scenario,
            "agent_count": 8,
            "start_index": 0,
            "seed": seed,
            "latency_steps": latency,
            "immediate_probability": probability,
        })

    assert len(rows) == 250
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="results/four_method_eval_matrix.csv",
    )
    args = parser.parse_args()

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        raise FileExistsError(
            f"{path} already exists; refusing to overwrite"
        )

    rows = build_matrix()

    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print("Evaluation configurations:", len(rows))
    print("Scenarios:", sorted({r["scenario_id"] for r in rows}))
    print("Seeds:", sorted({r["seed"] for r in rows}))
    print("Latencies:", sorted({r["latency_steps"] for r in rows}))
    print("Probabilities:", sorted({
        r["immediate_probability"] for r in rows
    }))
    print("Saved:", path)


if __name__ == "__main__":
    main()
