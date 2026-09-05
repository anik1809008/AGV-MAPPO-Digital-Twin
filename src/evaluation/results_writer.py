import csv
import os


RESULT_FIELDS = [
    "method",
    "scenario_id",
    "agent_count",
    "latency_steps",
    "m5_threshold",
    "success",
    "collision",
    "deadlock",
    "makespan",
    "path_length",
    "planning_time",
    "shield_interventions",
]





def append_result_csv(
    path,
    metrics,
):
    os.makedirs(
        os.path.dirname(path),
        exist_ok=True,
    )

    file_exists = os.path.exists(path)

    with open(
        path,
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=RESULT_FIELDS,
        )

        if not file_exists:
            writer.writeheader()


        writer.writerow(
            {
                field: metrics.get(
                    field,
                    "",
                )
                for field in RESULT_FIELDS
            }
        )
