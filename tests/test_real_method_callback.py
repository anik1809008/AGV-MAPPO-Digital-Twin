from src.evaluation.real_method_callback import run_real_method


def test_real_method_callback_imports():
    assert callable(run_real_method)
