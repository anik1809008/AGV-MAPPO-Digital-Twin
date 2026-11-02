"""Latency schedule for deterministic communication recovery experiments."""

from src.communication.latency_channel import FixedLatencyChannel


class ScheduledLatencyChannel(FixedLatencyChannel):
    def __init__(self, schedule):
        if not schedule or schedule[0][0] != 0:
            raise ValueError("Schedule must begin at timestep 0")

        if any(
            start < 0 or latency < 0
            for start, latency in schedule
        ):
            raise ValueError("Schedule values must be nonnegative")

        if any(
            schedule[i][0] >= schedule[i + 1][0]
            for i in range(len(schedule) - 1)
        ):
            raise ValueError("Schedule times must strictly increase")

        super().__init__(latency_steps=schedule[0][1])
        self.schedule = tuple(schedule)

    def send(self, message, current_timestep):
        for start, latency in reversed(self.schedule):
            if current_timestep >= start:
                self.latency_steps = latency
                break

        super().send(
            message,
            current_timestep=current_timestep,
        )
