from pathlib import Path

from src.environment.movingai_scenario import load_movingai_scenario


def test_load_movingai_scenario(tmp_path: Path):
    scenario_file = tmp_path / "test.scen"
    scenario_file.write_text(
        "version 1\n"
        "0\ttest.map\t5\t3\t0\t0\t4\t2\t6.0\n"
        "1\ttest.map\t5\t3\t4\t0\t0\t2\t6.0\n",
        encoding="utf-8",
    )

    scenarios = load_movingai_scenario(str(scenario_file))

    assert len(scenarios) == 2

    assert scenarios[0]["start"] == (0, 0)
    assert scenarios[0]["goal"] == (4, 2)
    assert scenarios[0]["optimal_length"] == 6.0

    assert scenarios[1]["start"] == (4, 0)
    assert scenarios[1]["goal"] == (0, 2)
