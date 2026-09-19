from src.evaluation.checkpoint_resolver import (
    resolve_evaluation_checkpoint,
    resolve_validation_checkpoint,
)
def test_m1_requires_no_checkpoint():
    path = resolve_evaluation_checkpoint(
        method="M1",
        agent_count=8,
        seed=0,
        scenario_id=1,
    )

    assert path is None


def test_m2_uses_m2_checkpoint():
    path = resolve_evaluation_checkpoint(
        method="M2",
        agent_count=8,
        seed=0,
        scenario_id=1,
    )

    assert path.endswith(
        "m2_agents8_seed0_scenario1.pt"
    )


def test_m4_uses_m3_checkpoint():
    path = resolve_evaluation_checkpoint(
        method="M4",
        agent_count=20,
        seed=2,
        scenario_id=4,
    )

    assert path.endswith(
        "m3_agents20_seed2_scenario4.pt"
    )


def test_m5_uses_m3_checkpoint():
    path = resolve_evaluation_checkpoint(
        method="M5",
        agent_count=8,
        seed=3,
        scenario_id=5,
    )

    assert path.endswith(
        "m3_agents8_seed3_scenario5.pt"
    )


def test_m6_uses_m6_checkpoint():
    path = resolve_evaluation_checkpoint(
        method="M6",
        agent_count=20,
        seed=4,
        scenario_id=6,
    )

    assert path.endswith(
        "m6_agents20_seed4_scenario6.pt"
    )
def test_validation_m1_requires_no_checkpoint():
    path = resolve_validation_checkpoint(
        method="M1",
        agent_count=8,
        seed=0,
        episodes_per_scenario=5,
    )

    assert path is None


def test_validation_m3_uses_budget_checkpoint():
    path = resolve_validation_checkpoint(
        method="M3",
        agent_count=8,
        seed=0,
        episodes_per_scenario=5,
    )

    assert path.endswith(
        "validation/m3_agents8_seed0_eps5.pt"
    )


def test_validation_m4_reuses_m3_checkpoint():
    path = resolve_validation_checkpoint(
        method="M4",
        agent_count=20,
        seed=2,
        episodes_per_scenario=10,
    )

    assert path.endswith(
        "validation/m3_agents20_seed2_eps10.pt"
    )
