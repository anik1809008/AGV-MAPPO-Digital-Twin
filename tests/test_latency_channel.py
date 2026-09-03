from src.communication.latency_channel import FixedLatencyChannel
from src.communication.telemetry import TelemetryMessage


def test_fixed_latency_delivery():
    channel = FixedLatencyChannel(latency_steps=2)

    msg = TelemetryMessage(
        agent_id=1,
        position=(8, 5),
        source_timestamp=20,
    )

    channel.send(msg, current_timestep=20)

    assert channel.receive_ready(20) == []
    assert channel.receive_ready(21) == []

    delivered = channel.receive_ready(22)

    assert delivered == [msg]


def test_zero_latency_delivery():
    channel = FixedLatencyChannel(latency_steps=0)

    msg = TelemetryMessage(
        agent_id=2,
        position=(4, 3),
        source_timestamp=10,
    )

    channel.send(msg, current_timestep=10)

    assert channel.receive_ready(10) == [msg]
