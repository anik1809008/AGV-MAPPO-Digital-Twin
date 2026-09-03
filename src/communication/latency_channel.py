import heapq


class FixedLatencyChannel:
    def __init__(self, latency_steps: int):
        if latency_steps < 0:
            raise ValueError("latency_steps must be >= 0")

        self.latency_steps = latency_steps
        self._queue = []
        self._counter = 0

    def send(self, message, current_timestep: int):
        delivery_timestep = current_timestep + self.latency_steps

        heapq.heappush(
            self._queue,
            (
                delivery_timestep,
                self._counter,
                message,
            ),
        )

        self._counter += 1

    def receive_ready(self, current_timestep: int):
        ready_messages = []

        while self._queue and self._queue[0][0] <= current_timestep:
            _, _, message = heapq.heappop(self._queue)
            ready_messages.append(message)

        return ready_messages
