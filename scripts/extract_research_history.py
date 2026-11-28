"""Read-only extraction of historical experimental evidence."""

import csv
from pathlib import Path
from collections import Counter
from statistics import mean

ROOT = Path("results")

TRUE_VALUES = {"true", "1", "1.0", "yes"}
FALSE_VALUES = {"false", "0", "0.0", "no"}


def boolean(value):
    value = str(value).strip().lower()

    if value in TRUE_VALUES:
        return True

    if value in FALSE_VALUES:
        return False

    return None


def numeric(rows, field):
    values = []

    for row in rows:
        try:
            values.append(float(row[field]))
        except (ValueError, TypeError, KeyError):
            continue

    return values


def summarize(path):
    print("\n" + "=" * 75)
    print("FILE:", path)

    try:
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fields = reader.fieldnames or []
    except Exception as exc:
        print("READ ERROR:", exc)
        return

    print("Rows:", len(rows))
    print("Columns:", ", ".join(fields))

    if not rows:
        return

    # Preserve configuration values.
    for field in (
        "method",
        "agent_count",
        "seed",
        "scenario_id",
        "latency_steps",
        "immediate_probability",
        "max_steps",
        "episodes_per_scenario",
        "training_budget",
        "threshold",
        "checkpoint",
        "checkpoint_source",
    ):
        if field in fields:
            counts = Counter(row[field] for row in rows)

            if len(counts) <= 20:
                print(f"{field}: {dict(counts)}")
            else:
                print(
                    f"{field}: "
                    f"{len(counts)} distinct values"
                )

    # Extract outcome counts without treating missing
    # or unrecognized values as failures.
    for field in (
        "success",
        "all_goals_reached",
        "collision",
        "deadlock",
        "timeout",
        "planning_failed",
    ):
        if field not in fields:
            continue

        values = [boolean(r[field]) for r in rows]
        valid = [v for v in values if v is not None]

        if valid:
            count = sum(valid)

            print(
                f"{field}: {count}/{len(valid)} "
                f"({100 * count / len(valid):.2f}%)"
            )

        if len(valid) != len(rows):
            print(
                f"  Missing/unrecognized {field}: "
                f"{len(rows) - len(valid)}"
            )

    success_field = next(
        (
            f for f in ("success", "all_goals_reached")
            if f in fields
        ),
        None,
    )

    if success_field:
        successful = [
            r for r in rows
            if boolean(r[success_field]) is True
        ]
    else:
        successful = []

    # Overall and successful-only efficiency.
    for field in (
        "makespan",
        "steps",
        "path_length",
        "planning_time",
        "shield_interventions",
        "return_replans",
        "successful_returns",
        "switch_count",
        "planning_failures",
    ):
        values = numeric(rows, field)

        if values:
            print(
                f"{field}: "
                f"mean={mean(values):.6f}, "
                f"min={min(values):.6f}, "
                f"max={max(values):.6f}, "
                f"N={len(values)}"
            )

        values = numeric(successful, field)

        if values and field in (
            "makespan", "steps",
            "path_length", "planning_time"
        ):
            print(
                f"  Successful-only {field}: "
                f"mean={mean(values):.6f}"
            )

    # Existing aggregate summaries: preserve raw values.
    if not success_field and len(rows) <= 30:
        print("RAW SUMMARY:")

        for row in rows:
            print(row)


def main():
    paths = sorted(ROOT.rglob("*.csv"))

    print("HISTORICAL EXPERIMENTAL EVIDENCE")
    print("CSV files discovered:", len(paths))

    for path in paths:
        summarize(path)

    print("\nEXTRACTION COMPLETED")


if __name__ == "__main__":
    main()
