import pytest
from src.evaluation.model_loader import (
    load_evaluation_models,
    load_validation_models,
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
def test_validation_m1_does_not_require_checkpoint():
    result = load_validation_models(
        method="M1",
        agent_count=8,
        seed=0,
        episodes_per_scenario=5,
    )

    assert result["checkpoint_path"] is None
    assert result["extra_state"] is None


def test_missing_validation_checkpoint_raises_error():
    with pytest.raises(
        FileNotFoundError,
        match="Checkpoint not found",
    ):
        load_validation_models(
            method="M6",
            agent_count=8,
            seed=999,
            episodes_per_scenario=999,
        )
def test_m4_validation_loader_uses_m3_selected_budget_checkpoint():
    from src.evaluation.checkpoint_resolver import (
        resolve_validation_checkpoint,
    )

    checkpoint_path = resolve_validation_checkpoint(
        method="M4",
        agent_count=8,
        seed=0,
        episodes_per_scenario=20,
    )

    assert checkpoint_path.endswith(
        "m3_agents8_seed0_eps20.pt"
    )
