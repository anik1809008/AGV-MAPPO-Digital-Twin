from pathlib import Path

from src.environment.movingai_map import load_movingai_map


def test_load_movingai_map(tmp_path: Path):
    map_file = tmp_path / "test.map"
    map_file.write_text(
        "type octile\n"
        "height 3\n"
        "width 5\n"
        "map\n"
        ".....\n"
        ".@@..\n"
        ".....\n",
        encoding="utf-8",
    )

    result = load_movingai_map(str(map_file))

    assert result["width"] == 5
    assert result["height"] == 3
    assert result["grid"] == [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ]
