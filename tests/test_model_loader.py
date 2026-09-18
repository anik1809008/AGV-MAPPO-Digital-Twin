import pytest

from src.evaluation.model_loader import (
    load_evaluation_models,
)


def test_m1_does_not_require_checkpoint():
    result = load_evaluation_models(
        method="M1",
        agent_count=8,
        seed=0,
        scenario_id=1,
    )

    assert result["checkpoint_path"] is None
    assert result["extra_state"] is None


def test_missing_checkpoint_raises_error():
    with pytest.raises(
        FileNotFoundError,
        match="Checkpoint not found",
    ):
        load_evaluation_models(
            method="M6",
            agent_count=8,
            seed=999,
            scenario_id=15,
        )
