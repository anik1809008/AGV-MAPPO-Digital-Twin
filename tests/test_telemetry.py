from src.communication.telemetry import TelemetryMessage


def test_telemetry_message_fields():
    msg = TelemetryMessage(
        agent_id=1,
        position=(8, 5),
        source_timestamp=20,
    )

    assert msg.agent_id == 1
    assert msg.position == (8, 5)
    assert msg.source_timestamp == 20
