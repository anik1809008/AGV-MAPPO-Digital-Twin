from pathlib import Path


def load_movingai_scenario(path: str):
    scenario_path = Path(path)

    with scenario_path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        raise ValueError("Scenario file is empty.")

    if not lines[0].startswith("version"):
        raise ValueError("Missing MovingAI scenario version header.")

    scenarios = []

    for line in lines[1:]:
        parts = line.split()

        if len(parts) < 9:
            raise ValueError(f"Invalid scenario row: {line}")

        scenarios.append(
            {
                "bucket": int(parts[0]),
                "map_name": parts[1],
                "map_width": int(parts[2]),
                "map_height": int(parts[3]),
                "start": (int(parts[4]), int(parts[5])),
                "goal": (int(parts[6]), int(parts[7])),
                "optimal_length": float(parts[8]),
            }
        )

    return scenarios
