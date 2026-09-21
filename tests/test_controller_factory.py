from src.baselines.stale_wait import ReachableSetWaitBaseline
from src.evaluation.controller_factory import (
    build_method_controllers,
)
from src.marl.networks import ActorNetwork
from src.safety.m4_controller import M4Controller


def build_actor():
    return ActorNetwork(
        input_dim=249,
        action_dim=5,
    )


def test_build_m4_controller():
    actor = build_actor()

    result = build_method_controllers(
        method="M4",
        actor=actor,
        grid=[
            [0, 0],
            [0, 0],
        ],
    )

    assert isinstance(
        result["m4_controller"],
        M4Controller,
    )

    assert result["m5_baseline"] is None


def test_build_m5_baseline():
    actor = build_actor()

    result = build_method_controllers(
        method="M5",
        actor=actor,
        grid=[
            [0, 0],
            [0, 0],
        ],
        m5_threshold=3,
    )

    assert result["m4_controller"] is None

    assert isinstance(
        result["m5_baseline"],
        ReachableSetWaitBaseline,
    )

    assert result["m5_baseline"].threshold == 3


def test_other_methods_need_no_extra_controller():
    actor = build_actor()

    for method in [
        "M1",
        "M2",
        "M3",
        "M6",
    ]:
        result = build_method_controllers(
            method=method,
            actor=actor,
            grid=[
                [0, 0],
                [0, 0],
            ],
        )

        assert result["m4_controller"] is None
        assert result["m5_baseline"] is None
