from dataclasses import dataclass
from typing import Tuple

Position = Tuple[int, int]


@dataclass(frozen=True)
class TelemetryMessage:
    agent_id: int
    position: Position
    source_timestamp: int
