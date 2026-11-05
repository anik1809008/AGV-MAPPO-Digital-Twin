"""Experimental LNS-SAT to M7 command handover."""

from src.hybrid.recovery_drain import drain_pending_commands


class ExecutionAwareHandover:
    def __init__(self):
        self.active = False
        self.failed = False
        self.resume_timestep = None

    def start(self):
        if self.active:
            raise ValueError("Handover already active")
        self.active = True
        self.failed = False
        self.resume_timestep = None

    def step(self, simulator, executor, timestep):
        if not self.active:
            raise ValueError("No active handover")

        # Issue no new commands while draining old commands.
        result = drain_pending_commands(
            simulator, executor, timestep
        )

        if result["collision"]:
            self.active = False
            self.failed = True
            return {
                **result,
                "ready": False,
                "failed": True,
            }

        ready = not result["pending_commands"]

        if ready:
            self.active = False
            # M7 must not act in the same timestep.
            self.resume_timestep = timestep + 1

        return {
            **result,
            "ready": ready,
            "failed": False,
        }

    def can_resume(self, timestep):
        return (
            not self.active
            and not self.failed
            and self.resume_timestep is not None
            and timestep >= self.resume_timestep
        )
