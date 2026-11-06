from unittest.mock import patch

import pytest

from src.marl.real_training_step import run_real_training_step


class StopAfterObservations(Exception):
    pass


class FakeSimulator:
    grid = [[0] * 7 for _ in range(7)]
    agent_positions = {0: (3, 1)}
    agent_goals = {0: (6, 1)}


@pytest.mark.parametrize("enabled", [False, True])
def test_dt_observation_mode_is_opt_in(enabled):
    simulator = FakeSimulator()

    kwargs = {
        "actor": None,
        "critic": None,
        "simulator": simulator,
        "method": "M7",
        "trusted_positions": {0: (1, 1)},
        "reachable_occupancies": {0: {(1, 1), (2, 1), (3, 1)}},
        "aoi_values": {0: 2},
        "use_dt_observations": enabled,
    }

    with patch(
        "src.marl.real_training_step.build_all_agent_inputs"
    ) as legacy, patch(
        "src.marl.real_training_step.build_dt_only_agent_inputs"
    ) as dt_only, patch(
        "src.marl.real_training_step.collect_single_step",
        side_effect=StopAfterObservations,
    ):
        with pytest.raises(StopAfterObservations):
            run_real_training_step(**kwargs)

        if enabled:
            dt_only.assert_called_once()
            legacy.assert_not_called()
            assert dt_only.call_args.kwargs["trusted_positions"] == {
                0: (1, 1)
            }
        else:
            legacy.assert_called_once()
            dt_only.assert_not_called()
            assert legacy.call_args.kwargs["agent_positions"] == [
                (3, 1)
            ]


def test_dt_observation_mode_rejects_non_m7():
    with pytest.raises(ValueError, match="require M7"):
        run_real_training_step(
            actor=None,
            critic=None,
            simulator=FakeSimulator(),
            method="M3",
            trusted_positions={0: (1, 1)},
            reachable_occupancies={0: {(1, 1)}},
            aoi_values={0: 0},
            use_dt_observations=True,
        )
