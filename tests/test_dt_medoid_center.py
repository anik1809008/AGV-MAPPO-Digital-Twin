import pytest

from src.marl.dt_medoid_center import select_dt_medoid_centers


def test_singleton():
    result = select_dt_medoid_centers(
        {0: (0, 0)},
        {0: {(4, 0)}},
    )
    assert result[0] == (4, 0)


def test_medoid_of_uncertainty_set():
    result = select_dt_medoid_centers(
        {0: (0, 0)},
        {0: {(1, 0), (2, 0), (3, 0)}},
    )
    assert result[0] == (2, 0)


def test_selected_center_is_plausible():
    possible = {(2, 1), (3, 1), (3, 2), (5, 2)}
    result = select_dt_medoid_centers(
        {0: (0, 1)},
        {0: possible},
    )
    assert result[0] in possible


def test_empty_set_rejected():
    with pytest.raises(ValueError, match="Missing possible positions"):
        select_dt_medoid_centers({0: (0, 0)}, {0: set()})
