from pathlib import Path


FREE_CELLS = {".", "G", "S"}
BLOCKED_CELLS = {"@", "O", "T", "W"}


def load_movingai_map(path: str):
    map_path = Path(path)

    with map_path.open("r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    if len(lines) < 4:
        raise ValueError("Invalid MovingAI map file.")

    if not lines[0].startswith("type"):
        raise ValueError("Missing MovingAI map type header.")

    height = int(lines[1].split()[1])
    width = int(lines[2].split()[1])

    if lines[3].strip() != "map":
        raise ValueError("Missing 'map' marker.")

    grid_lines = lines[4:]

    if len(grid_lines) != height:
        raise ValueError(
            f"Expected {height} map rows, found {len(grid_lines)}."
        )

    grid = []

    for row in grid_lines:
        if len(row) != width:
            raise ValueError(
                f"Expected row width {width}, found {len(row)}."
            )

        parsed_row = []

        for cell in row:
            if cell in FREE_CELLS:
                parsed_row.append(0)
            elif cell in BLOCKED_CELLS:
                parsed_row.append(1)
            else:
                raise ValueError(f"Unknown map symbol: {cell}")

        grid.append(parsed_row)

    return {
        "width": width,
        "height": height,
        "grid": grid,
    }
