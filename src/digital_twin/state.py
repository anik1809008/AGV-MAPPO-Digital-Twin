from dataclasses import dataclass, field
from typing import List, Tuple

Position = Tuple[int, int]


@dataclass
class AgentTwinState:
    agent_id: int
    last_trusted_position: Position
    last_trusted_timestamp: int
    goal: Position

    command_history: List[int] = field(default_factory=list)

    def age_of_information(self, current_timestep: int) -> int:
        return current_timestep - self.last_trusted_timestamp
