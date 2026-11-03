"""Paired evaluation of Original Hybrid and Recovery Hybrid."""

import csv
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

ORIGINAL = Path("results/eval250_two_way.csv")
RECOVERY = Path("results/eval250_recovery_hybrid.csv")
OUTPUT = Path("results/hybrid_paired_diagnostics.csv")

MATCH_FIELDS = (
    "scenario_id",
    "seed",
    "start_index",
    "latency_steps",
    "immediate_probability",
)


def yes(value):
    return str(value).strip().lower() == "true"


def load(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = [r["episode_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate episode IDs in {path}")
    return {r["episode_id"]: r for r in rows}


def exact_mcnemar(b, c):
    """Two-sided exact McNemar test."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1))
    return min(1.0, 2 * tail / (2 ** n))


def paired_bootstrap_ci(differences, repeats=5000, seed=2026):
    rng = random.Random(seed)
    n = len(differences)
    estimates = sorted(
        sum(rng.choices(differences, k=n)) / n
        for _ in range(repeats)
    )
    return estimates[int(0.025 * repeats)], estimates[int(0.975 * repeats)]


def analyze(original, recovery):
    if set(original) != set(recovery):
        raise ValueError("Episode IDs do not match")

    if len(original) != 250:
        raise ValueError("Expected exactly 250 paired episodes")

    details = []
    for episode_id in sorted(original, key=int):
        a = original[episode_id]
        b = recovery[episode_id]

        for field in MATCH_FIELDS:
            av, bv = a[field], b[field]
            if field == "immediate_probability":
                equal = float(av) == float(bv)
            else:
                equal = av == bv
            if not equal:
                raise ValueError(
                    f"Episode {episode_id}: unmatched {field}: {av} vs {bv}"
                )

        original_success = yes(a["success"])
        recovery_success = yes(b["success"])
        attempts = int(b["recovery_attempts"] or 0)

        details.append({
            "episode_id": episode_id,
            **{field: a[field] for field in MATCH_FIELDS},
            "original_success": original_success,
            "recovery_success": recovery_success,
            "original_collision": yes(a["collision"]),
            "recovery_collision": yes(b["collision"]),
            "original_deadlock": yes(a["deadlock"]),
            "recovery_deadlock": yes(b["deadlock"]),
            "original_steps": a["makespan"],
            "recovery_steps": b["makespan"],
            "recovery_attempts": attempts,
            "recovery_pause_steps": int(b["recovery_pause_steps"] or 0),
            "successful_returns": int(b["successful_returns"] or 0),
            "difference": int(recovery_success) - int(original_success),
        })

    return details


def main():
    details = analyze(load(ORIGINAL), load(RECOVERY))
    count = len(details)
    wins = sum(r["difference"] == 1 for r in details)
    losses = sum(r["difference"] == -1 for r in details)
    differences = [r["difference"] for r in details]
    ci = paired_bootstrap_ci(differences)

    print("PAIRED HYBRID EVALUATION")
    print("Matched episodes:", count)
    print("Recovery wins:", wins)
    print("Recovery losses:", losses)
    print("Equal success outcomes:", count - wins - losses)
    print("Success difference (Recovery - Original):",
          f"{100 * sum(differences) / count:+.2f} percentage points")
    print("Paired bootstrap 95% CI:",
          f"[{100 * ci[0]:+.2f}, {100 * ci[1]:+.2f}] pp")
    print("Exact McNemar p-value:",
          f"{exact_mcnemar(losses, wins):.6f}")

    activated = [r for r in details if r["recovery_attempts"] > 0]
    print("\nRECOVERY ACTIVITY")
    print("Episodes with activation:", len(activated))
    print("Recovery attempts:", sum(r["recovery_attempts"] for r in details))
    print("Pause steps:", sum(r["recovery_pause_steps"] for r in details))
    print("Recorded successful returns:",
          sum(r["successful_returns"] for r in details))
    print("Activated episodes: wins=",
          sum(r["difference"] == 1 for r in activated),
          "losses=", sum(r["difference"] == -1 for r in activated))

    print("\nPER-SCENARIO SUCCESS COUNTS")
    for scenario in sorted({r["scenario_id"] for r in details}):
        group = [r for r in details if r["scenario_id"] == scenario]
        print(
            f"Scenario {scenario}: "
            f"Original={sum(r['original_success'] for r in group)}/{len(group)}, "
            f"Recovery={sum(r['recovery_success'] for r in group)}/{len(group)}, "
            f"Activated={sum(r['recovery_attempts'] > 0 for r in group)}"
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)

    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()
