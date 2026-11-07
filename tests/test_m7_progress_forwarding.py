from unittest.mock import patch

from src.safety.m7_controller import M7Controller


def test_m7_forwards_progress_information():
    controller = M7Controller(
        actor=object(),
        shield=object(),
        enable_progress_forwarding=True,
    )

    ranked = [(1, 0.8), (0, 0.2)]

    with patch(
        "src.safety.m7_controller.rank_actions_by_probability",
        return_value=ranked,
    ), patch.object(
        controller.action_filter,
        "select_safe_action",
        return_value=(1, 0.8),
    ) as filter_mock:

        controller.select_action(
            observation_vector=[0.0],
            agent_id=0,
            possible_current_positions={(5, 5)},
            other_current_positions={},
            other_next_positions={},
            reachable_occupancies={},
            current_position=(5, 5),
            goal_position=(8, 5),
            prefer_progress=True,
        )

    kwargs = filter_mock.call_args.kwargs

    assert kwargs["current_position"] == (5, 5)
    assert kwargs["goal_position"] == (8, 5)
    assert kwargs["prefer_progress"] is True
