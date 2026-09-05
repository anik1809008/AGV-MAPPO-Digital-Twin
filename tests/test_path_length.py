from src.evaluation.path_length import compute_step_path_length


def test_compute_step_path_length():
    previous_positions = {
        0: (0, 0),
        1: (2, 1),
    }

    current_positions = {
        0: (1, 0),
        1: (2, 1),
    }

    assert compute_step_path_length(
        previous_positions,
        current_positions,
    ) == 1
