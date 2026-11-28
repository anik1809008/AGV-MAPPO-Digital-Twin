"""Final matched statistical analysis of DICR recovery."""

import csv
from pathlib import Path

import numpy as np

import analyze_step11_ablation as previous

from analyze_hybrid_evaluation import (
    load,
    yes,
    exact_mcnemar,
)

from analyze_step11_ablation import (
    validate,
    success_array,
    cluster_arrays,
    cluster_bootstrap_ci,
    cluster_sign_flip,
    numeric_array,
    holm_adjust,
)

FILES = {
    **previous.FILES,
    "dicr": Path("results/step13_dicr_final1000.csv"),
}

NAMES = {
    **previous.NAMES,
    "dicr": "Combined Recovery + DICR",
}

OUTPUT = Path("results/step14_dicr_statistics.csv")


def compare(data, ids, outcomes, a, b, label, selected):
    selected_ids = [
        k for k, include in zip(ids, selected)
        if include
    ]

    differences = (
        outcomes[b][selected] - outcomes[a][selected]
    )

    wins = int(np.sum(differences == 1))
    losses = int(np.sum(differences == -1))
    ties = int(np.sum(differences == 0))

    route_sums, route_sizes = cluster_arrays(
        data["base"],
        selected_ids,
        differences,
        ("scenario_id", "start_index"),
    )

    scenario_sums, scenario_sizes = cluster_arrays(
        data["base"],
        selected_ids,
        differences,
        ("scenario_id",),
    )

    ci = 100 * cluster_bootstrap_ci(
        route_sums, route_sizes
    )

    route_p, route_method = cluster_sign_flip(
        route_sums
    )

    scenario_p, scenario_method = cluster_sign_flip(
        scenario_sums
    )

    old_success = int(outcomes[a][selected].sum())
    new_success = int(outcomes[b][selected].sum())

    return {
        "comparison": label,
        "reference": a,
        "proposed": b,
        "episodes": len(selected_ids),
        "reference_success": old_success,
        "proposed_success": new_success,
        "rescued": wins,
        "regressed": losses,
        "ties": ties,
        "effect_pp": 100 * differences.mean(),
        "mcnemar_p": exact_mcnemar(wins, losses),
        "route_ci_low_pp": ci[0],
        "route_ci_high_pp": ci[1],
        "route_clusters": len(route_sums),
        "nonzero_route_clusters": int(
            np.count_nonzero(route_sums)
        ),
        "route_cluster_p": route_p,
        "route_test_method": route_method,
        "scenario_clusters": len(scenario_sums),
        "nonzero_scenario_clusters": int(
            np.count_nonzero(scenario_sums)
        ),
        "scenario_cluster_p": scenario_p,
        "scenario_test_method": scenario_method,
    }


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

    latency = np.array([
        int(data["base"][k]["latency_steps"])
        for k in ids
    ])

    probability = np.array([
        float(data["base"][k]["immediate_probability"])
        for k in ids
    ])

    all_episodes = np.ones(len(ids), dtype=bool)

    perfect = (
        (latency == 0)
        & (probability == 1.0)
    )

    imperfect = ~perfect

    deterministic_delayed = (
        np.isin(latency, [1, 2])
        & (probability == 1.0)
    )

    stochastic = probability < 1.0

    print("\n=== FINAL FIVE-METHOD PERFORMANCE ===")

    for name, rows in data.items():
        success = int(outcomes[name].sum())
        collisions = sum(
            yes(r["collision"]) for r in rows.values()
        )
        deadlocks = sum(
            yes(r["deadlock"]) for r in rows.values()
        )

        print(
            f"{NAMES[name]}: "
            f"{success}/1000 ({success / 10:.1f}%), "
            f"collisions={collisions}, "
            f"deadlocks={deadlocks}"
        )

    # Four overlapping research comparisons.
    comparisons = [
        (
            "base", "dicr",
            "Original Hybrid vs Final DICR",
            all_episodes,
        ),
        (
            "both", "dicr",
            "Combined Recovery vs DICR",
            all_episodes,
        ),
        (
            "both", "dicr",
            "All imperfect conditions",
            imperfect,
        ),
        (
            "both", "dicr",
            "Deterministic delayed conditions",
            deterministic_delayed,
        ),
    ]

    results = [
        compare(
            data, ids, outcomes,
            a, b, label, selected,
        )
        for a, b, label, selected in comparisons
    ]

    for field in (
        "mcnemar_p",
        "route_cluster_p",
        "scenario_cluster_p",
    ):
        holm_adjust(results, field)

    print("\n=== STATISTICAL COMPARISONS ===")

    for result in results:
        print("\n" + result["comparison"])

        n = result["episodes"]

        print(f"  Episodes: {n}")
        print(
            f"  Success: "
            f"{result['reference_success']}/{n} -> "
            f"{result['proposed_success']}/{n}"
        )
        print(f"  Rescued: {result['rescued']}")
        print(f"  Regressed: {result['regressed']}")
        print(f"  Effect: {result['effect_pp']:+.2f} pp")

        print(
            f"  Route-cluster 95% CI: "
            f"[{result['route_ci_low_pp']:+.2f}, "
            f"{result['route_ci_high_pp']:+.2f}] pp"
        )

        print(
            f"  McNemar p: "
            f"{result['mcnemar_p']:.8g}"
        )

        print(
            f"  Route-cluster p: "
            f"{result['route_cluster_p']:.8g}"
        )

        print(
            f"  Scenario-cluster p: "
            f"{result['scenario_cluster_p']:.8g}"
        )

        print(
            f"  Holm-adjusted route p: "
            f"{result['route_cluster_p_holm']:.8g}"
        )

        print(
            f"  Holm-adjusted scenario p: "
            f"{result['scenario_cluster_p_holm']:.8g}"
        )

        print(
            f"  Nonzero route clusters: "
            f"{result['nonzero_route_clusters']}/"
            f"{result['route_clusters']}"
        )

        print(
            f"  Nonzero scenario clusters: "
            f"{result['nonzero_scenario_clusters']}/"
            f"{result['scenario_clusters']}"
        )

    print("\n=== CONDITION-SPECIFIC RESULTS ===")

    for label, mask in (
        ("Perfect", perfect),
        ("Imperfect", imperfect),
        ("Deterministic delayed", deterministic_delayed),
        ("Stochastic execution", stochastic),
    ):
        n = int(mask.sum())

        old = int(outcomes["both"][mask].sum())
        new = int(outcomes["dicr"][mask].sum())

        print(
            f"{label}: "
            f"{old}/{n} -> {new}/{n}, "
            f"{100 * old / n:.2f}% -> "
            f"{100 * new / n:.2f}%"
        )

    print("\n=== PER-SCENARIO RESULTS ===")

    scenarios = sorted({
        data["base"][k]["scenario_id"]
        for k in ids
    })

    for scenario in scenarios:
        mask = np.array([
            data["base"][k]["scenario_id"] == scenario
            for k in ids
        ])

        n = int(mask.sum())

        old = int(outcomes["both"][mask].sum())
        new = int(outcomes["dicr"][mask].sum())

        print(
            f"Scenario {scenario}: "
            f"{old}/{n} -> {new}/{n}, "
            f"gain={100 * (new-old)/n:+.2f} pp"
        )

    print("\n=== PAIRED EFFICIENCY TRADE-OFF ===")

    shared = [
        k for i, k in enumerate(ids)
        if outcomes["both"][i]
        and outcomes["dicr"][i]
    ]

    print("Common successful episodes:", len(shared))

    for field in (
        "makespan",
        "path_length",
        "planning_time",
    ):
        before = numeric_array(
            data["both"], shared, field
        )

        after = numeric_array(
            data["dicr"], shared, field
        )

        delta = after - before

        print(f"\n{field}")
        print(f"  Before: {before.mean():.4f}")
        print(f"  After: {after.mean():.4f}")
        print(f"  Mean difference: {delta.mean():+.4f}")

        if before.mean() != 0:
            print(
                f"  Relative change: "
                f"{100 * delta.mean() / before.mean():+.2f}%"
            )

        print(f"  Median difference: {np.median(delta):.4f}")
        print(f"  Improved: {int(np.sum(delta < 0))}")
        print(f"  Unchanged: {int(np.sum(delta == 0))}")
        print(f"  Worsened: {int(np.sum(delta > 0))}")

    print("\n=== PLANNING TIME DISTRIBUTION ===")

    for name in ("base", "both", "dicr"):
        timing = numeric_array(
            data[name], ids, "planning_time"
        )

        print(f"\n{NAMES[name]}")
        print(f"  Mean: {timing.mean():.4f} s")
        print(f"  Median: {np.median(timing):.4f} s")
        print(f"  P95: {np.percentile(timing, 95):.4f} s")
        print(f"  P99: {np.percentile(timing, 99):.4f} s")
        print(f"  Maximum: {timing.max():.4f} s")

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
