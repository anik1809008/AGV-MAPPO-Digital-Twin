"""Bounded recovery-pause state, separate from AGV control."""

class RecoveryPause:
    def __init__(self, max_pause_steps=3, cooldown_steps=20):
        if max_pause_steps < 1 or cooldown_steps < 1:
            raise ValueError("Recovery limits must be positive")
        self.max_pause_steps = max_pause_steps
        self.cooldown_steps = cooldown_steps
        self.active = False
        self.started_at = None
        self.next_allowed_at = 0

    def start(self, timestep):
        if self.active or timestep < self.next_allowed_at:
            return False
        self.active = True
        self.started_at = timestep
        return True

    def expired(self, timestep):
        return (
            self.active
            and timestep - self.started_at >= self.max_pause_steps
        )

    def finish(self, timestep):
        if not self.active:
            raise ValueError("Recovery pause is not active")
        self.active = False
        self.started_at = None
        self.next_allowed_at = timestep + self.cooldown_steps
