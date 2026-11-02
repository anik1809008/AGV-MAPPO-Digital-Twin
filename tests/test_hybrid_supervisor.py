from src.hybrid.supervisor import HybridSupervisor


def test_stays_normal_with_fresh_information():
    supervisor = HybridSupervisor(aoi_threshold=2)
    assert supervisor.observe(timestep=0, aoi_values={0: 0, 1: 0}) is None
    assert supervisor.mode == "LNS-SAT"


def test_switches_on_aoi_without_truth_data():
    supervisor = HybridSupervisor(aoi_threshold=2)
    event = supervisor.observe(timestep=3, aoi_values={0: 0, 1: 2})
    assert event.reason == "aoi_threshold_exceeded"
    assert event.agent_id == 1
    assert event.timestep == 3
    assert supervisor.mode == "M7"
    assert len(supervisor.events) == 1
    assert supervisor.observe(timestep=4, aoi_values={0: 0, 1: 3}) is None
    assert len(supervisor.events) == 1


def test_switches_on_stale_occupancy_conflict():
    supervisor = HybridSupervisor(aoi_threshold=5)
    event = supervisor.observe(
        timestep=4,
        aoi_values={0: 0, 1: 1},
        planned_next_positions={0: (4, 2), 1: (8, 2)},
        possible_current_positions={
            0: {(2, 2)},
            1: {(4, 2), (5, 2)},
        },
    )
    assert event.reason == "stale_occupancy_conflict"
    assert event.agent_id == 1
    assert supervisor.mode == "M7"


def test_rejects_invalid_threshold():
    try:
        HybridSupervisor(aoi_threshold=0)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")
