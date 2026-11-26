
"""Matched statistical analysis of Step 11 recovery ablations."""

import csv
import math
from itertools import product
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze_hybrid_evaluation import (
    MATCH_FIELDS,
    yes,
    load,
    exact_mcnemar,
)


FILES = {
    "base": Path("results/step8_final1000_baseline.csv"),
    "single": Path("results/step11_single_only1000.csv"),
    "two": Path("results/step11_two_only1000.csv"),
    "both": Path("results/step8_final1000_improved.csv"),
}

NAMES = {
    "base": "Original Two-Way",
    "single": "Single-AGV Only",
    "two": "Two-AGV Only",
    "both": "Combined Recovery",
}

COMPARISONS = [
    ("base", "single"),
    ("base", "two"),
    ("base", "both"),
    ("single", "two"),
    ("single", "both"),
    ("two", "both"),
]

OUTPUT = Path("results/step12_ablation_statistics.csv")

BOOTSTRAP_REPEATS = 10000
PERMUTATION_REPEATS = 100000
RANDOM_SEED = 2026


def validate(data):
    base = data["base"]

    assert len(base) == 1000

    for name, rows in data.items():
        assert len(rows) == 1000, name
        assert set(rows) == set(base), name

        for episode_id, row in rows.items():
            reference = base[episode_id]

            for field in MATCH_FIELDS:
                if field == "immediate_probability":
                    equal = (
                        float(row[field])
                        == float(reference[field])
                    )
                else:
                    equal = row[field] == reference[field]

                if not equal:
                    raise ValueError(
                        f"{name}: episode {episode_id}, "
                        f"unmatched {field}"
                    )

    print("Validation passed: 1000 matched episodes.")
    print("All experimental configuration fields match.")


def success_array(rows, ids):
    return np.array(
        [int(yes(rows[k]["success"])) for k in ids],
        dtype=int,
    )


def cluster_arrays(base, ids, differences, fields):
    groups = defaultdict(list)

    for episode_id, difference in zip(ids, differences):
        key = tuple(base[episode_id][f] for f in fields)
        groups[key].append(int(difference))

    sums = np.array(
        [sum(values) for values in groups.values()]
    )
    sizes = np.array(
        [len(values) for values in groups.values()]
    )

    return sums, sizes


def cluster_bootstrap_ci(sums, sizes):
    rng = np.random.default_rng(RANDOM_SEED)
    n = len(sums)

    estimates = []

    for _ in range(BOOTSTRAP_REPEATS):
        selected = rng.integers(0, n, size=n)

        estimate = (
            sums[selected].sum()
            / sizes[selected].sum()
        )
        estimates.append(estimate)

    return np.percentile(estimates, [2.5, 97.5])


def cluster_sign_flip(sums):
    observed = abs(int(sums.sum()))

    # Zero-effect clusters do not affect sign permutations.
    sums = sums[sums != 0]
    n = len(sums)

    if n == 0:
        return 1.0, "exact"

    if n <= 16:
        extreme = 0

        for signs in product((-1, 1), repeat=n):
            statistic = abs(
                sum(s * x for s, x in zip(signs, sums))
            )
            extreme += statistic >= observed

        return extreme / (2 ** n), "exact"

    rng = np.random.default_rng(RANDOM_SEED)
    extreme = 0

    for _ in range(PERMUTATION_REPEATS):
        signs = rng.choice((-1, 1), size=n)
        statistic = abs(np.dot(signs, sums))
        extreme += statistic >= observed

    p = (extreme + 1) / (PERMUTATION_REPEATS + 1)

    return p, "Monte Carlo"


def numeric_array(rows, ids, field):
    values = np.array(
        [float(rows[k][field]) for k in ids],
        dtype=float,
    )

    if not np.isfinite(values).all():
        raise ValueError(f"Invalid numeric values: {field}")

    return values


def print_summary(data, ids, outcomes):
    print("\n=== OVERALL PERFORMANCE ===")

    for name, rows in data.items():
        success = int(outcomes[name].sum())
        collisions = sum(yes(r["collision"]) for r in rows.values())
        deadlocks = sum(yes(r["deadlock"]) for r in rows.values())

        incomplete_other = sum(
            not yes(r["success"])
            and not yes(r["collision"])
            and not yes(r["deadlock"])
            for r in rows.values()
        )

        successful_ids = [
            k for k in ids if yes(rows[k]["success"])
        ]

        timing = numeric_array(rows, ids, "planning_time")

        print(f"\n{NAMES[name]}")
        print(f"  Success: {success}/1000 ({success / 10:.1f}%)")
        print(f"  Collisions: {collisions}")
        print(f"  Deadlocks: {deadlocks}")
        print(f"  Other incomplete episodes: {incomplete_other}")

        if successful_ids:
            for field in ("makespan", "path_length"):
                values = numeric_array(
                    rows, successful_ids, field
                )
                print(
                    f"  Mean successful {field}: "
                    f"{values.mean():.3f}"
                )

        print(f"  Mean planning time: {timing.mean():.3f} s")
        print(f"  Median planning time: {np.median(timing):.3f} s")
        print(f"  P95 planning time: {np.percentile(timing, 95):.3f} s")
        print(f"  P99 planning time: {np.percentile(timing, 99):.3f} s")
        print(f"  Max planning time: {timing.max():.3f} s")


def paired_analysis(data, ids, outcomes):
    print("\n=== PAIRED STATISTICAL COMPARISONS ===")

    results = []

    for a, b in COMPARISONS:
        differences = outcomes[b] - outcomes[a]

        wins = int(np.sum(differences == 1))
        losses = int(np.sum(differences == -1))
        ties = len(ids) - wins - losses
        effect = 100 * differences.mean()

        route_sums, route_sizes = cluster_arrays(
            data["base"],
            ids,
            differences,
            ("scenario_id", "start_index"),
        )

        scenario_sums, scenario_sizes = cluster_arrays(
            data["base"],
            ids,
            differences,
            ("scenario_id",),
        )

        ci = 100 * cluster_bootstrap_ci(
            route_sums, route_sizes
        )

        route_p, route_method = cluster_sign_flip(route_sums)
        scenario_p, scenario_method = cluster_sign_flip(
            scenario_sums
        )

        mcnemar_p = exact_mcnemar(wins, losses)

        print(f"\n{NAMES[a]} -> {NAMES[b]}")
        print(f"  Rescued: {wins}")
        print(f"  Regressed: {losses}")
        print(f"  Ties: {ties}")
        print(f"  Success difference: {effect:+.2f} pp")
        print(f"  Exact McNemar p: {mcnemar_p:.8g}")

        print(
            f"  Route-cluster 95% CI: "
            f"[{ci[0]:+.2f}, {ci[1]:+.2f}] pp"
        )

        print(f"  Route clusters: {len(route_sums)}")
        print(f"  Route-cluster p ({route_method}): {route_p:.8g}")
        print(f"  Scenario clusters: {len(scenario_sums)}")
        print(
            f"  Scenario-cluster p ({scenario_method}): "
            f"{scenario_p:.8g}"
        )

        results.append({
            "reference": a,
            "comparison": b,
            "rescued": wins,
            "regressed": losses,
            "ties": ties,
            "effect_pp": effect,
            "mcnemar_p": mcnemar_p,
            "route_ci_low_pp": ci[0],
            "route_ci_high_pp": ci[1],
            "route_clusters": len(route_sums),
            "route_cluster_p": route_p,
            "route_test_method": route_method,
            "scenario_clusters": len(scenario_sums),
            "scenario_cluster_p": scenario_p,
            "scenario_test_method": scenario_method,
        })

    return results


def recovery_overlap(outcomes):
    print("\n=== RECOVERY OVERLAP ===")

    base = outcomes["base"].astype(bool)
    single = outcomes["single"].astype(bool)
    two = outcomes["two"].astype(bool)
    both = outcomes["both"].astype(bool)

    s = ~base & single
    t = ~base & two
    c = ~base & both

    print("Single-AGV rescues:", int(s.sum()))
    print("Two-AGV rescues:", int(t.sum()))
    print("Rescued by both individual methods:", int((s & t).sum()))
    print("Single-AGV exclusive rescues:", int((s & ~t).sum()))
    print("Two-AGV exclusive rescues:", int((t & ~s).sum()))
    print("Union of individual rescues:", int((s | t).sum()))
    print("Combined Recovery rescues:", int(c.sum()))

    print(
        "Combined-only rescues:",
        int((c & ~(s | t)).sum()),
    )

    print(
        "Individual rescues lost by Combined:",
        int(((s | t) & ~both).sum()),
    )


def paired_efficiency(data, ids, outcomes):
    print("\n=== PAIRED EFFICIENCY ===")

    for a, b in COMPARISONS:
        shared_success = [
            k for i, k in enumerate(ids)
            if outcomes[a][i] and outcomes[b][i]
        ]

        print(f"\n{NAMES[a]} -> {NAMES[b]}")
        print(f"  Jointly successful episodes: {len(shared_success)}")

        for field in (
            "makespan",
            "path_length",
            "planning_time",
        ):
            if not shared_success:
                continue

            before = numeric_array(
                data[a], shared_success, field
            )
            after = numeric_array(
                data[b], shared_success, field
            )

            difference = after - before

            print(
                f"  {field}: "
                f"{before.mean():.3f} -> {after.mean():.3f}, "
                f"mean paired difference={difference.mean():+.3f}"
            )

        before = numeric_array(data[a], ids, "planning_time")
        after = numeric_array(data[b], ids, "planning_time")

        print(
            "  All-episode planning time difference: "
            f"{(after - before).mean():+.3f} s"
        )


def grouped_performance(data, ids, outcomes, fields):
    groups = defaultdict(list)

    for i, k in enumerate(ids):
        row = data["base"][k]
        key = tuple(row[field] for field in fields)
        groups[key].append(i)

    for key, indices in sorted(groups.items()):
        print(f"\n{key}: N={len(indices)}")

        for name in FILES:
            wins = int(outcomes[name][indices].sum())
            print(
                f"  {NAMES[name]:22s}: "
                f"{wins}/{len(indices)} "
                f"({100 * wins / len(indices):.1f}%)"
            )


def holm_adjust(results, field):
    ordered = sorted(
        range(len(results)),
        key=lambda i: results[i][field],
    )

    previous = 0.0
    total = len(ordered)

    for rank, i in enumerate(ordered):
        adjusted = min(
            1.0,
            (total - rank) * results[i][field],
        )
        previous = max(previous, adjusted)
        results[i][field + "_holm"] = previous


def main():
    data = {
        name: load(path)
        for name, path in FILES.items()
    }

    validate(data)

    ids = sorted(data["base"], key=int)

    outcomes = {
        name: success_array(rows, ids)
        for name, rows in data.items()
    }

    print_summary(data, ids, outcomes)

    results = paired_analysis(data, ids, outcomes)

    recovery_overlap(outcomes)
    paired_efficiency(data, ids, outcomes)

    print("\n=== PER-SCENARIO PERFORMANCE ===")
    grouped_performance(
        data, ids, outcomes, ("scenario_id",)
    )

    print("\n=== PER-UNCERTAINTY PERFORMANCE ===")
    grouped_performance(
        data,
        ids,
        outcomes,
        ("latency_steps", "immediate_probability"),
    )

    for field in (
        "mcnemar_p",
        "route_cluster_p",
        "scenario_cluster_p",
    ):
        holm_adjust(results, field)

    print("\n=== HOLM-ADJUSTED P-VALUES ===")

    for row in results:
        print(
            f"{row['reference']} -> {row['comparison']}: "
            f"McNemar={row['mcnemar_p_holm']:.8g}, "
            f"Route={row['route_cluster_p_holm']:.8g}, "
            f"Scenario={row['scenario_cluster_p_holm']:.8g}"
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(results[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()
