from src.evaluation.policy_mapping import (
    get_policy_method,
)


def test_policy_mapping():
    assert get_policy_method("M1") is None

    assert get_policy_method("M2") == "M2"
    assert get_policy_method("M3") == "M3"

    assert get_policy_method("M4") == "M3"
    assert get_policy_method("M5") == "M3"

    assert get_policy_method("M6") == "M6"


def test_policy_mapping_rejects_unknown_method():
    try:
        get_policy_method("M7")
    except ValueError as exc:
        assert (
            "Unknown evaluation method"
            in str(exc)
        )
    else:
        raise AssertionError(
            "Expected ValueError"
        )
