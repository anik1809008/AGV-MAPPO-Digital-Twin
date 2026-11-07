from unittest.mock import patch

from src.safety.m7_controller import M7Controller


def call_controller(controller):
    return controller.select_action(
        observation_vector=[0.0],
        agent_id=0,
        possible_current_positions={(1, 1)},
        other_current_positions={},
        other_next_positions={},
        reachable_occupancies={},
        goal_position=(6, 1),
    )


def test_robust_mode_selects_improving_action():
    controller = M7Controller(
        actor=object(),
        shield=object(),
        enable_robust_progress=True,
    )

    with patch(
        "src.safety.m7_controller.rank_actions_by_probability",
        return_value=[(0, 0.8), (3, 0.2)],
    ), patch(
        "src.safety.m7_controller.select_robust_progress_action",
        return_value=(3, 0.2),
    ) as robust, patch.object(
        controller.action_filter,
        "select_safe_action",
    ) as ordinary:

        result = call_controller(controller)

    assert result == (3, 0.2)
    robust.assert_called_once()
    ordinary.assert_not_called()


def test_default_mode_preserves_original_filter():
    controller = M7Controller(
        actor=object(),
        shield=object(),
    )

    with patch(
        "src.safety.m7_controller.rank_actions_by_probability",
        return_value=[(0, 0.8), (3, 0.2)],
    ), patch(
        "src.safety.m7_controller.select_robust_progress_action",
    ) as robust, patch.object(
        controller.action_filter,
        "select_safe_action",
        return_value=(0, 0.8),
    ) as ordinary:

        result = call_controller(controller)

    assert result == (0, 0.8)
    robust.assert_not_called()
    ordinary.assert_called_once()


def test_robust_mode_falls_back_to_original_filter():
    controller = M7Controller(
        actor=object(),
        shield=object(),
        enable_robust_progress=True,
    )

    with patch(
        "src.safety.m7_controller.rank_actions_by_probability",
        return_value=[(0, 0.8)],
    ), patch(
        "src.safety.m7_controller.select_robust_progress_action",
        return_value=None,
    ), patch.object(
        controller.action_filter,
        "select_safe_action",
        return_value=(0, 0.8),
    ) as ordinary:

        result = call_controller(controller)

    assert result == (0, 0.8)
    ordinary.assert_called_once()
