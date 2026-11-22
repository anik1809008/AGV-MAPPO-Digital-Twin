from src.hybrid.recovery_route_guard import (
    build_expected_positions,
    recovery_route_is_valid,
)


def make_case():
    starts = {0: (0, 0), 1: (2, 1)}
    actions = {
        0: [3, 3],
        1: [0, 0],
    }

    expected = build_expected_positions(starts, actions, 2)

    plan = {"recovery_expected_positions": expected}

    dt = {
        "possible_current_positions": {
            0: {(1, 0)},
            1: {(2, 1)},
        },
        "aoi_values": {0: 0, 1: 0},
    }

    return plan, dt


def test_expected_route_positions():
    plan, _ = make_case()

    assert plan["recovery_expected_positions"] == [
        {0: (0, 0), 1: (2, 1)},
        {0: (1, 0), 1: (2, 1)},
        {0: (2, 0), 1: (2, 1)},
    ]


def test_accepts_correct_dt_positions():
    plan, dt = make_case()

    assert recovery_route_is_valid(plan, dt, 1, False)


def test_rejects_unexpected_position():
    plan, dt = make_case()
    dt["possible_current_positions"][0] = {(0, 0)}

    assert not recovery_route_is_valid(plan, dt, 1, False)


def test_rejects_stale_telemetry():
    plan, dt = make_case()
    dt["aoi_values"][0] = 1

    assert not recovery_route_is_valid(plan, dt, 1, False)


def test_rejects_ambiguous_position():
    plan, dt = make_case()
    dt["possible_current_positions"][0] = {
        (1, 0), (0, 0)
    }

    assert not recovery_route_is_valid(plan, dt, 1, False)


def test_rejects_pending_commands():
    plan, dt = make_case()

    assert not recovery_route_is_valid(plan, dt, 1, True)


def test_rejects_incomplete_actions():
    try:
        build_expected_positions(
            {0: (0, 0), 1: (2, 1)},
            {0: [3]},
            1,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Incomplete actions accepted")
