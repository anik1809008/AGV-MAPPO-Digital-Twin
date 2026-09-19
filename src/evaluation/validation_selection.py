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
