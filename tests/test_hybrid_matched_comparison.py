from scripts.run_hybrid_matched_comparison import trial_schedule, normalize_lns


def test_schedule_is_distinct_and_matches_existing_benchmark():
    schedule = trial_schedule(10)
    assert len(schedule) == len(set(schedule)) == 10
    assert schedule[:5] == [(16, 0), (17, 0), (18, 0), (19, 0), (20, 0)]
    assert schedule[5:] == [(16, 8), (17, 8), (18, 8), (19, 8), (20, 8)]


def test_failed_lns_has_no_misleading_zero_makespan():
    result = normalize_lns({'planning_failed': True})
    assert result['planning_failed'] is True
    assert result['success'] is False
    assert result['makespan'] == ''
    assert result['planning_time'] == ''
