import pytest

from src.evaluation.marl_method_runner import run_marl_method


def test_invalid_marl_method_raises_error():
    with pytest.raises(
        ValueError,
        match="method must be one of: M2, M3, M4, M5",
    ):
        run_marl_method(
            method="M6",
            actor=None,
            critic=None,
            simulator=None,
            multi_agent_buffer=None,
        )
