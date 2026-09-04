import pytest
from src.marl.reward import compute_reward


def test_progress_reward():
    reward = compute_reward(
        previous_distance=5,
        current_distance=4,
    )

    assert reward == 0.08


def test_goal_reward():
    reward = compute_reward(
        reached_goal=True,
        previous_distance=1,
        current_distance=0,
    )

    assert reward == 10.08


def test_collision_penalty():
    reward = compute_reward(
        collision=True,
    )

    assert reward == -10.02


def test_deadlock_penalty():
    reward = compute_reward(
        deadlock=True,
    )

    assert reward == -5.02


def test_moving_away_penalty():
    reward = compute_reward(
        previous_distance=4,
        current_distance=5,
    )

    assert reward == pytest.approx(-0.12)
