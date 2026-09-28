from src.safety.progress_monitor import ProgressMonitor


def test_progress_monitor_detects_stagnation():
    grid = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]

    monitor = ProgressMonitor(
        grid=grid,
        stagnation_steps=3,
    )

    goal = (3, 0)

    assert monitor.update(0, (0, 0), goal) is False
    assert monitor.update(0, (1, 0), goal) is False
    assert monitor.update(0, (1, 1), goal) is False
    assert monitor.update(0, (0, 1), goal) is False
    assert monitor.update(0, (1, 1), goal) is True
