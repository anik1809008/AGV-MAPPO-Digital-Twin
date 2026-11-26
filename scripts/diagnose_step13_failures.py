
"""Diagnostic replay of selected frozen Step 8 failed episodes."""

import csv
import json
from collections import defaultdict, Counter
from pathlib import Path
from unittest.mock import patch

import scripts.run_four_method_evaluation as evaluation
import src.hybrid.two_way_episode_runner as hybrid

from src.evaluation.model_loader import load_validation_models


MATRIX = Path("results/step8_final1000_matrix.csv")
REFERENCE = Path("results/step8_final1000_improved.csv")
OUTPUT = Path("results/step13_failure_diagnostics.jsonl")

MAX_STEPS = 400
EPISODES_PER_CONDITION = 2


def yes(value):
    return str(value).strip().lower() == "true"


def load_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def select_episodes(reference):
    groups = defaultdict(list)

    for row in reference:
        if yes(row["success"]):
            continue

        key = (
            int(row["latency_steps"]),
            float(row["immediate_probability"]),
        )
        groups[key].append(row)

    selected = []

    for key, rows in sorted(groups.items()):
        rows.sort(key=lambda r: int(r["episode_id"]))

        deadlocks = [
            r for r in rows if yes(r["deadlock"])
        ]
        others = [
            r for r in rows if not yes(r["deadlock"])
        ]

        group = []

        # Include one deadlock when available.
        if deadlocks:
            group.append(deadlocks[0])

        # Fill remaining slots with non-deadlock failures.
        group.extend(
            others[:EPISODES_PER_CONDITION - len(group)]
        )

        if len(group) < EPISODES_PER_CONDITION:
            group.extend(
                r for r in deadlocks
                if r not in group
            )

        selected.extend(group[:EPISODES_PER_CONDITION])

    return selected


def diagnose(spec, reference, models):
    original_runner = evaluation.run_two_way_episode
    original_dt = hybrid.get_digital_twin_inputs

    observations = {}
    captured = {}

    def observe_dt(*args, **kwargs):
        dt = original_dt(*args, **kwargs)

        observations["timestep"] = kwargs.get(
            "current_timestep"
        )

        observations["aoi"] = {
            str(a): int(age)
            for a, age in dt["aoi_values"].items()
        }

        observations["possible_positions"] = {
            str(a): sorted(
                [list(p) for p in positions]
            )
            for a, positions in
            dt["possible_current_positions"].items()
        }

        return dt

    def observe_runner(*args, **kwargs):
        context = kwargs["context"]
        supervisor = kwargs["supervisor"]

        with patch.object(
            hybrid,
            "get_digital_twin_inputs",
            side_effect=observe_dt,
        ):
            metrics = original_runner(*args, **kwargs)

        sim = context["simulator"]
        executor = context["delayed_executor"]

        agents = []

        for agent_id in sorted(sim.agent_goals):
            position = tuple(sim.agent_positions[agent_id])
            goal = tuple(sim.agent_goals[agent_id])

            distance = sim.navigation_distance(
                position, goal
            )

            agents.append({
                "agent_id": int(agent_id),
                "final_position": list(position),
                "goal": list(goal),
                "finished": position == goal,
                "remaining_distance": (
                    None if distance is None
                    else float(distance)
                ),
                "last_pre_action_aoi": observations.get(
                    "aoi", {}
                ).get(str(agent_id)),
                "last_pre_action_position_set": (
                    observations.get(
                        "possible_positions", {}
                    ).get(str(agent_id), [])
                ),
                "pending_commands": len(
                    executor.pending_commands.get(
                        agent_id, []
                    )
                ),
            })

        unfinished = [
            a for a in agents if not a["finished"]
        ]

        captured.update({
            "final_mode": supervisor.mode,
            "unfinished_count": len(unfinished),
            "unfinished_agents": [
                a["agent_id"] for a in unfinished
            ],
            "non_progress_steps": int(
                sim.non_progress_steps
            ),
            "last_dt_timestep": observations.get(
                "timestep"
            ),
            "agents": agents,
            "return_replans": metrics["return_replans"],
            "planning_failures": metrics[
                "planning_failures"
            ],
            "lns_steps": metrics["lns_steps"],
            "m7_steps": metrics["m7_steps"],
        })

        return metrics

    with patch.object(
        evaluation,
        "run_two_way_episode",
        side_effect=observe_runner,
    ):
        outcome = evaluation.run_episode(
            method="DT-ROBUST-HYBRID",
            specification=spec,
            models=models,
            max_steps=MAX_STEPS,
            single_agv_completion=True,
            two_agv_completion=True,
        )

    # Validate replay against the frozen outcome.
    mismatches = []

    for field in ("success", "collision", "deadlock"):
        if bool(outcome[field]) != yes(reference[field]):
            mismatches.append(field)

    for field in (
        "makespan",
        "path_length",
        "return_replans",
        "successful_returns",
        "switch_count",
        "planning_failures",
    ):
        if float(outcome[field]) != float(reference[field]):
            mismatches.append(field)

    return {
        "episode_id": int(spec["episode_id"]),
        "scenario_id": int(spec["scenario_id"]),
        "start_index": int(spec["start_index"]),
        "seed": int(spec["seed"]),
        "latency_steps": int(spec["latency_steps"]),
        "immediate_probability": float(
            spec["immediate_probability"]
        ),
        "success": bool(outcome["success"]),
        "collision": bool(outcome["collision"]),
        "deadlock": bool(outcome["deadlock"]),
        "makespan": int(outcome["makespan"]),
        "replay_matches_reference": not mismatches,
        "replay_mismatches": mismatches,
        **captured,
    }


def main():
    matrix_rows = load_csv(MATRIX)
    reference = load_csv(REFERENCE)

    matrix = {
        r["episode_id"]: r for r in matrix_rows
    }

    assert len(matrix_rows) == len(matrix) == 1000
    assert len(reference) == 1000

    reference_ids = {
        r["episode_id"] for r in reference
    }

    assert len(reference_ids) == 1000
    assert reference_ids == set(matrix)

    match_fields = (
        "scenario_id",
        "start_index",
        "seed",
        "latency_steps",
        "immediate_probability",
    )

    for row in reference:
        spec = matrix[row["episode_id"]]

        for field in match_fields:
            if float(row[field]) != float(spec[field]):
                raise ValueError(
                    f"Episode {row['episode_id']}: "
                    f"unmatched {field}"
                )

    selected = select_episodes(reference)

    print("STEP 13 FAILURE DIAGNOSTIC PILOT")
    print("Selected episodes:", len(selected))
    print("Horizon:", MAX_STEPS)

    print("\nSELECTED EPISODE IDS")
    for row in selected:
        print(
            row["episode_id"],
            row["latency_steps"],
            row["immediate_probability"],
            "deadlock=" + row["deadlock"],
        )

    model_cache = {}
    records = []

    OUTPUT.parent.mkdir(
        parents=True, exist_ok=True
    )

    with OUTPUT.open("w") as handle:
        for index, row in enumerate(selected, 1):
            spec = matrix[row["episode_id"]]

            key = (
                int(spec["agent_count"]),
                int(spec["seed"]),
            )

            if key not in model_cache:
                model_cache[key] = load_validation_models(
                    method="M7",
                    agent_count=key[0],
                    seed=key[1],
                    episodes_per_scenario=20,
                )

            print(
                f"\n[{index}/{len(selected)}] "
                f"Episode {spec['episode_id']}",
                flush=True,
            )

            record = diagnose(
                spec, row, model_cache[key]
            )

            records.append(record)

            handle.write(json.dumps(record) + "\n")
            handle.flush()

            print(
                "Condition:",
                record["latency_steps"],
                record["immediate_probability"],
            )

            print(
                "Replay matches frozen reference:",
                record["replay_matches_reference"],
            )

            if record["replay_mismatches"]:
                print(
                    "Mismatches:",
                    record["replay_mismatches"],
                )

            print("Final mode:", record["final_mode"])
            print("Steps:", record["makespan"])
            print("Deadlock:", record["deadlock"])

            print(
                "Unfinished AGVs:",
                record["unfinished_count"],
            )

            print(
                "Unfinished IDs:",
                record["unfinished_agents"],
            )

            print(
                "Non-progress steps:",
                record["non_progress_steps"],
            )

            print(
                "Return replans:",
                record["return_replans"],
            )

            for agent in record["agents"]:
                if agent["finished"]:
                    continue

                print(
                    f"  AGV {agent['agent_id']}: "
                    f"position={agent['final_position']}, "
                    f"goal={agent['goal']}, "
                    f"distance={agent['remaining_distance']}, "
                    f"last_AoI={agent['last_pre_action_aoi']}, "
                    f"last_set_size="
                    f"{len(agent['last_pre_action_position_set'])}, "
                    f"pending={agent['pending_commands']}"
                )

    matched = [
        r for r in records
        if r["replay_matches_reference"]
    ]

    counts = Counter(
        r["unfinished_count"] for r in matched
    )

    print("\n=== MATCHED PILOT SUMMARY ===")
    print("Selected episodes:", len(records))
    print("Matched replays:", len(matched))
    print("Unmatched replays:", len(records) - len(matched))

    for count, frequency in sorted(counts.items()):
        print(
            f"{count} unfinished AGV(s): "
            f"{frequency} episodes"
        )

    print(
        "One or two unfinished AGVs:",
        sum(
            r["unfinished_count"] in (1, 2)
            for r in matched
        ),
    )

    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()
