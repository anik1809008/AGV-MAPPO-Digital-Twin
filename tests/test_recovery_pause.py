import pytest

from src.hybrid.recovery_pause import RecoveryPause


def test_pause_starts_and_expires():
    pause = RecoveryPause(max_pause_steps=3)
    assert pause.start(10)
    assert not pause.expired(12)
    assert pause.expired(13)


def test_cooldown_prevents_immediate_restart():
    pause = RecoveryPause(max_pause_steps=3, cooldown_steps=20)
    assert pause.start(10)
    pause.finish(12)
    assert not pause.start(13)
    assert not pause.start(31)
    assert pause.start(32)


def test_cannot_start_twice():
    pause = RecoveryPause()
    assert pause.start(10)
    assert not pause.start(11)


def test_finish_requires_active_pause():
    with pytest.raises(ValueError):
        RecoveryPause().finish(0)


def test_invalid_limits():
    with pytest.raises(ValueError):
        RecoveryPause(max_pause_steps=0)
    with pytest.raises(ValueError):
        RecoveryPause(cooldown_steps=0)
