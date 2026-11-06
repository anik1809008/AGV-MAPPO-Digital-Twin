from unittest.mock import Mock, patch

import pytest

from src.marl.real_training_step import run_real_training_step
from src.safety.shield import SafetyShield


class Simulator:
    grid = [[0] * 7]
    agent_positions = {0: (4, 0), 1: (6, 0)}
    agent_goals = {0: (6, 0), 1: (0, 0)}


def run_dt_step(possible):
    simulator = Simulator()

    controller = Mock()
    controller.shield = SafetyShield(simulator.grid)
    controller.select_action.return_value = (0, 1.0)

    monitor = Mock()
    monitor.is_stagnating.return_value = False

    with patch(
        "src.marl.real_training_step.collect_single_step",
        return_value={
            "actions": [0, 0],
            "log_probs": [0, 0],
            "value": 0,
            "centralized_state": [],
        },
    ), patch(
        "src.marl.real_training_step.execute_training_step",
        return_value={
            "collision": False,
            "rewards": [0, 0],
            "dones": [False, False],
            "current_positions": [(4, 0), (6, 0)],
        },
    ):
        run_real_training_step(
            actor=None,
            critic=None,
            simulator=simulator,
            method="M7",
            trusted_positions={0: (1, 0), 1: (5, 0)},
            reachable_occupancies={
                0: {(1, 0), (2, 0)},
                1: {(5, 0)},
            },
            possible_current_positions=possible,
            aoi_values={0: 2, 1: 0},
            m7_controller=controller,
            m7_progress_monitor=monitor,
            use_dt_observations=True,
        )

    return controller, monitor


def test_dt_controller_does_not_receive_true_position():
    controller, monitor = run_dt_step({
        0: {(1, 0), (2, 0)},
        1: {(5, 0)},
    })

    first = controller.select_action.call_args_list[0].kwargs

    assert first["current_position"] == (1, 0)
    assert first["possible_current_positions"] == {
        (1, 0), (2, 0)
    }
    assert first["current_position"] != (4, 0)
    assert first["other_next_positions"] == {}

    # Stale telemetry must not update the progress monitor.
    monitor.update.assert_called_once_with(
        agent_id=1,
        position=(5, 0),
        goal=(0, 0),
    )


def test_previous_agent_reservations_use_uncertainty_sets():
    controller, _ = run_dt_step({
        0: {(1, 0), (2, 0)},
        1: {(5, 0)},
    })

    second = controller.select_action.call_args_list[1].kwargs

    assert {(1, 0), (2, 0)}.issubset(
        second["reachable_occupancies"][0]
    )


def test_missing_possible_positions_are_rejected():
    with pytest.raises(ValueError, match="Missing possible positions"):
        run_dt_step({
            0: set(),
            1: {(5, 0)},
        })
