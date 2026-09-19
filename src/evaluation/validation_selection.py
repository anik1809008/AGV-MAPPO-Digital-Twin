import csv
def summarize_validation_group(rows):
    if not rows:
        raise ValueError(
            "Validation group must not be empty"
        )

    count = len(rows)

    collision_rate = sum(
        bool(row["collision"])
        for row in rows
    ) / count

    deadlock_rate = sum(
        bool(row["deadlock"])
        for row in rows
    ) / count

    success_rate = sum(
        bool(row["success"])
        for row in rows
    ) / count

    mean_makespan = sum(
        float(row["makespan"])
        for row in rows
    ) / count

    mean_path_length = sum(
        float(row["path_length"])
        for row in rows
    ) / count

    mean_planning_time = sum(
        float(row["planning_time"])
        for row in rows
    ) / count

    return {
        "collision_rate": collision_rate,
        "deadlock_rate": deadlock_rate,
        "success_rate": success_rate,
        "mean_makespan": mean_makespan,
        "mean_path_length": mean_path_length,
        "mean_planning_time": mean_planning_time,
    }


def validation_selection_key(summary):
    return (
        summary["collision_rate"],
        summary["deadlock_rate"],
        -summary["success_rate"],
        summary["mean_makespan"],
        summary["mean_path_length"],
        summary["mean_planning_time"],
    )


def select_best_validation_candidate(
    candidate_rows,
):
    if not candidate_rows:
        raise ValueError(
            "No validation candidates provided"
        )

    best_candidate = None
    best_summary = None
    best_key = None

    for candidate, rows in candidate_rows.items():
        summary = summarize_validation_group(
            rows
        )

        key = validation_selection_key(
            summary
        )

        if best_key is None or key < best_key:
            best_candidate = candidate
            best_summary = summary
            best_key = key

    return {
        "candidate": best_candidate,
        "summary": best_summary,
    }
def parse_csv_bool(value):
    normalized = str(value).strip().lower()

    if normalized == "true":
        return True

    if normalized == "false":
        return False

    raise ValueError(
        f"Invalid boolean value: {value}"
    )


def load_validation_rows(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        rows = list(
            csv.DictReader(file)
        )

    parsed_rows = []

    for row in rows:
        parsed = dict(row)

        parsed["collision"] = parse_csv_bool(
            row["collision"]
        )
        parsed["deadlock"] = parse_csv_bool(
            row["deadlock"]
        )
        parsed["success"] = parse_csv_bool(
            row["success"]
        )

        parsed["makespan"] = float(
            row["makespan"]
        )
        parsed["path_length"] = float(
            row["path_length"]
        )
        parsed["planning_time"] = float(
            row["planning_time"]
        )

        if row["episodes_per_scenario"]:
            parsed["episodes_per_scenario"] = int(
                row["episodes_per_scenario"]
            )
        else:
            parsed["episodes_per_scenario"] = None

        if row["m5_threshold"]:
            parsed["m5_threshold"] = int(
                row["m5_threshold"]
            )
        else:
            parsed["m5_threshold"] = None

        parsed["scenario_id"] = int(
            row["scenario_id"]
        )
        parsed["agent_count"] = int(
            row["agent_count"]
        )
        parsed["seed"] = int(
            row["seed"]
        )

        parsed_rows.append(parsed)

    return parsed_rows


def group_validation_candidates(
    rows,
    method,
    agent_count,
    seed,
):
    grouped = {}

    for row in rows:
        if row["method"] != method:
            continue

        if row["agent_count"] != agent_count:
            continue

        if row["seed"] != seed:
            continue

        scenario_id = row["scenario_id"]

        if scenario_id < 16 or scenario_id > 20:
            raise ValueError(
                "Validation selection received "
                "a non-validation scenario"
            )

        budget = row["episodes_per_scenario"]

        if method == "M5":
            candidate = (
                budget,
                row["m5_threshold"],
            )
        else:
            candidate = budget

        grouped.setdefault(
            candidate,
            [],
        ).append(row)

    return grouped
