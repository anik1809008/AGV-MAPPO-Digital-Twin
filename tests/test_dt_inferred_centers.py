import numpy as np
import pytest

from src.marl.dt_only_observation import build_dt_only_agent_inputs
from src.marl.dt_position_center import select_dt_observation_centers
from src.marl.real_training_step import run_real_training_step


def test_inferred_center_changes_observation():
    data = dict(
        grid=[[0] * 8 for _ in range(8)],
        agent_goals={0: (7, 7)},
        trusted_positions={0: (1, 1)},
        reachable_occupancies={0: {(1, 1), (3, 1)}},
        aoi_values={0: 2},
    )

    centers = select_dt_observation_centers(
        data["trusted_positions"],
        {0: {(3, 1)}},
    )

    original = build_dt_only_agent_inputs(**data)
    inferred = build_dt_only_agent_inputs(
        **data, observation_centers=centers
    )

    assert centers[0] == (3, 1)
    assert not np.array_equal(original[0], inferred[0])


def test_inferred_mode_requires_dt_observations():
    class Simulator:
        agent_positions = {0: (2, 1)}
        agent_goals = {0: (7, 1)}
        grid = [[0] * 8]

    with pytest.raises(ValueError, match="require DT observations"):
        run_real_training_step(
            actor=None,
            critic=None,
            simulator=Simulator(),
            method="M7",
            trusted_positions={0: (1, 1)},
            reachable_occupancies={0: {(1, 1)}},
            aoi_values={0: 1},
            use_dt_inferred_centers=True,
        )
