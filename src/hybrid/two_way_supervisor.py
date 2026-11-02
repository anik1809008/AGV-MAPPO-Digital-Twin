"""Two-way switching decisions; AGV execution is handled separately."""

from src.hybrid.supervisor import HybridSupervisor, SwitchEvent


class TwoWayHybridSupervisor(HybridSupervisor):
    def __init__(
        self,
        aoi_threshold=2,
        recovery_steps=3,
        min_dwell_steps=3,
    ):
        super().__init__(aoi_threshold=aoi_threshold)

        if recovery_steps < 1 or min_dwell_steps < 1:
            raise ValueError("Steps must be >= 1")

        self.recovery_steps = recovery_steps
        self.min_dwell_steps = min_dwell_steps
        self.reliable_steps = 0
        self.last_switch_timestep = None

    def observe(
        self,
        *,
        timestep,
        aoi_values,
        planned_next_positions=None,
        possible_current_positions=None,
        pending_commands=False,
    ):
        if self.mode == "LNS-SAT":
            event = super().observe(
                timestep=timestep,
                aoi_values=aoi_values,
                planned_next_positions=planned_next_positions,
                possible_current_positions=possible_current_positions,
            )
            if event is not None:
                self.last_switch_timestep = timestep
                self.reliable_steps = 0
            return event

        # Returning to classical planning needs trustworthy state.
        known_agents = set(aoi_values)
        positions = possible_current_positions or {}

        reliable = (
            bool(known_agents)
            and known_agents == set(positions)
            and all(age == 0 for age in aoi_values.values())
            and all(len(positions[agent_id]) == 1
                    for agent_id in known_agents)
            and not pending_commands
        )

        if not reliable:
            self.reliable_steps = 0
            return None

        self.reliable_steps += 1

        dwell_satisfied = (
            self.last_switch_timestep is not None
            and timestep - self.last_switch_timestep
            >= self.min_dwell_steps
        )

        if (
            self.reliable_steps >= self.recovery_steps
            and dwell_satisfied
        ):
            return "RETURN_READY"

        return None

    def confirm_return(self, timestep):
        """Call only after verified-state replanning succeeds."""
        if self.mode != "M7":
            raise ValueError("Return requires M7 mode")

        if self.reliable_steps < self.recovery_steps:
            raise ValueError("Reliable recovery not established")

        if (
            self.last_switch_timestep is None
            or timestep - self.last_switch_timestep
            < self.min_dwell_steps
        ):
            raise ValueError("Minimum dwell time not satisfied")

        event = SwitchEvent(
            timestep=timestep,
            previous_mode="M7",
            new_mode="LNS-SAT",
            reason="reliable_state_and_replan",
            agent_id=-1,
            aoi=0,
        )
        self.mode = "LNS-SAT"
        self.events.append(event)
        self.last_switch_timestep = timestep
        self.reliable_steps = 0
        return event

    def fallback_to_m7(self, timestep):
        """Record a classical planning failure."""
        if self.mode != "LNS-SAT":
            raise ValueError("Fallback requires LNS-SAT mode")

        event = self._switch(
            timestep=timestep,
            reason="lns_planning_failed",
            agent_id=-1,
            aoi=0,
        )
        self.last_switch_timestep = timestep
        self.reliable_steps = 0
        return event
