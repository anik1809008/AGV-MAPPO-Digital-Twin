from dataclasses import dataclass


@dataclass(frozen=True)
class SwitchEvent:
    timestep: int
    previous_mode: str
    new_mode: str
    reason: str
    agent_id: int
    aoi: int


class HybridSupervisor:
    """One-way, DT-only supervisor for an initial hybrid demonstration.

    This selects a controller; it does not execute or safety-check actions.
    """

    def __init__(self, aoi_threshold=2):
        if aoi_threshold < 1:
            raise ValueError("aoi_threshold must be >= 1")
        self.aoi_threshold = aoi_threshold
        self.mode = "LNS-SAT"
        self.events = []

    def observe(
        self,
        *,
        timestep,
        aoi_values,
        planned_next_positions=None,
        possible_current_positions=None,
    ):
        if self.mode == "M7":
            return None

        planned_next_positions = planned_next_positions or {}
        possible_current_positions = possible_current_positions or {}

        for moving_id, next_cell in sorted(planned_next_positions.items()):
            for other_id, possible_cells in sorted(possible_current_positions.items()):
                if moving_id == other_id:
                    continue
                if aoi_values.get(other_id, 0) > 0 and next_cell in possible_cells:
                    return self._switch(
                        timestep, "stale_occupancy_conflict",
                        other_id, aoi_values[other_id],
                    )

        for agent_id, age in sorted(aoi_values.items()):
            if age >= self.aoi_threshold:
                return self._switch(
                    timestep, "aoi_threshold_exceeded", agent_id, age,
                )
        return None

    def _switch(self, timestep, reason, agent_id, aoi):
        event = SwitchEvent(
            timestep=timestep,
            previous_mode=self.mode,
            new_mode="M7",
            reason=reason,
            agent_id=agent_id,
            aoi=aoi,
        )
        self.mode = "M7"
        self.events.append(event)
        return event
