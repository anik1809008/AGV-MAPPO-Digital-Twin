"""Experimental execution-aware Hybrid supervisor.

Preserves the existing TwoWayHybridSupervisor baseline.
Uses pending command state, not simulator ground-truth positions.
"""

from src.hybrid.two_way_supervisor import TwoWayHybridSupervisor


class ExecutionAwareTwoWaySupervisor(TwoWayHybridSupervisor):

    def observe(
        self,
        *,
        timestep,
        aoi_values,
        planned_next_positions=None,
        possible_current_positions=None,
        pending_commands=False,
    ):
        if self.mode == "LNS-SAT" and pending_commands:
            event = self._switch(
                timestep=timestep,
                reason="pending_execution_uncertainty",
                agent_id=-1,
                aoi=max(aoi_values.values(), default=0),
            )
            self.last_switch_timestep = timestep
            self.reliable_steps = 0
            return event

        return super().observe(
            timestep=timestep,
            aoi_values=aoi_values,
            planned_next_positions=planned_next_positions,
            possible_current_positions=possible_current_positions,
            pending_commands=pending_commands,
        )
