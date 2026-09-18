POLICY_METHOD_BY_EVALUATION_METHOD = {
    "M1": None,
    "M2": "M2",
    "M3": "M3",
    "M4": "M3",
    "M5": "M3",
    "M6": "M6",
}


def get_policy_method(method):
    if method not in POLICY_METHOD_BY_EVALUATION_METHOD:
        raise ValueError(
            f"Unknown evaluation method: {method}"
        )

    return POLICY_METHOD_BY_EVALUATION_METHOD[
        method
    ]
