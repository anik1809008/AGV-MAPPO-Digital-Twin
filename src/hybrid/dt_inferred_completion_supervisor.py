"""Experimental supervisor for DT-inferred completion recovery."""

from src.hybrid.execution_aware_supervisor import (
    ExecutionAwareTwoWaySupervisor,
)
from src.hybrid.supervisor import SwitchEvent


class DTInferredCompletionSupervisor(
    ExecutionAwareTwoWaySupervisor
):
    """Preserve normal switching and support guarded DICR."""

    def __init__(
        self,
        aoi_threshold=2,
        recovery_steps=3,
        min_dwell_steps=3,
    ):
        super().__init__(
            aoi_threshold=aoi_threshold,
            recovery_steps=recovery_steps,
            min_dwell_steps=min_dwell_steps,
        )

        self.inferred_completion_active = False

    def confirm_inferred_completion(
        self,
        timestep,
        *,
        eligibility_confirmed,
        route_valid,
        aoi,
    ):
        """Activate only after runner eligibility and route checks."""

        if self.mode != "M7":
            raise ValueError("DICR activation requires M7 mode")

        if eligibility_confirmed is not True:
            raise ValueError("DICR eligibility not confirmed")

        if route_valid is not True:
            raise ValueError("DICR route not validated")

        if not 1 <= aoi <= 2:
            raise ValueError("DICR requires bounded positive AoI")

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
            reason="dt_inferred_completion",
            agent_id=-1,
            aoi=aoi,
        )

        self.mode = "LNS-SAT"
        self.events.append(event)
        self.last_switch_timestep = timestep
        self.reliable_steps = 0
        self.inferred_completion_active = True

        return event

    def observe(
        self,
        *,
        timestep,
        aoi_values,
        planned_next_positions=None,
        possible_current_positions=None,
        pending_commands=False,
        inferred_route_valid=None,
    ):
        if self.inferred_completion_active:
            # Active DICR requires explicit validation on every step.
            # Missing validation is treated as a failure.
            if (
                self.mode == "LNS-SAT"
                and inferred_route_valid is True
                and not pending_commands
            ):
                return None

            event = self._switch(
                timestep=timestep,
                reason="dt_inferred_route_invalid",
                agent_id=-1,
                aoi=max(aoi_values.values(), default=0),
            )

            self.inferred_completion_active = False
            self.last_switch_timestep = timestep
            self.reliable_steps = 0

            return event

        # Original execution-aware supervisor behavior.
        return super().observe(
            timestep=timestep,
            aoi_values=aoi_values,
            planned_next_positions=planned_next_positions,
            possible_current_positions=possible_current_positions,
            pending_commands=pending_commands,
        )

    def fallback_to_m7(self, timestep):
        event = super().fallback_to_m7(timestep)

        self.inferred_completion_active = False

        return event
