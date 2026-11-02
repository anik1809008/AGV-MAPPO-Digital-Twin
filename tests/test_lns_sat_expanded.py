import csv

import pytest

from scripts.run_lns_sat_expanded import CSV_FIELDS, make_trials, run_benchmark


def test_fifty_trials_are_unique_nonoverlapping():
    trials = make_trials(50)
    assert len(trials) == 50
    assert len(set(trials)) == 50
    assert set(s for s, _ in trials) == {16, 17, 18, 19, 20}
    for scenario_id in range(16, 21):
        indices = sorted(start for s, start in trials if s == scenario_id)
        assert indices == list(range(0, 80, 8))
    assert "start_index" in CSV_FIELDS
    assert "planning_failed" in CSV_FIELDS


def test_small_benchmark_writes_identifiable_rows(tmp_path, monkeypatch):
    import scripts.run_lns_sat_expanded as benchmark

    def fake_evaluate(scenario_id, start_index, max_iterations):
        return ({"success": True, "collision": False, "planning_failed": False,
                 "method": "LNS-SAT", "scenario_id": scenario_id,
                 "start_index": start_index, "agent_count": 8,
                 "makespan": 42}, (scenario_id, start_index))

    monkeypatch.setattr(benchmark, "evaluate_trial", fake_evaluate)
    path = tmp_path / "test.csv"
    run_benchmark(2, path)
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert [r["scenario_id"] for r in rows] == ["16", "17"]
    with pytest.raises(FileExistsError):
        run_benchmark(2, path)
