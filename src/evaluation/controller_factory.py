from src.safety.m7_controller import M7Controller
from src.baselines.stale_wait import ReachableSetWaitBaseline
from src.safety.m4_controller import M4Controller
from src.safety.shield import SafetyShield
from src.safety.progress_monitor import ProgressMonitor
from src.safety.local_recovery import LocalRecoveryPlanner
def build_method_controllers(
    method,
    actor,
    grid,
    m5_threshold=2,
):
    m4_controller = None
    m7_controller = None
    m5_baseline = None
    m7_progress_monitor = None
    m7_recovery_planner = None
    if method == "M4":
        shield = SafetyShield(grid)

        m4_controller = M4Controller(
            actor=actor,
            shield=shield,
        )

    if method == "M7":
        shield = SafetyShield(grid)

        m7_controller = M7Controller(
            actor=actor,
            shield=shield,
        )

        m7_progress_monitor = ProgressMonitor(
            grid=grid,
            stagnation_steps=10,
        )
        m7_recovery_planner = LocalRecoveryPlanner(
            grid=grid,
        )
    if method == "M5":
        m5_baseline = ReachableSetWaitBaseline(
            threshold=m5_threshold,
        )
    return {
        "m4_controller": m4_controller,
        "m5_baseline": m5_baseline,
        "m7_progress_monitor": m7_progress_monitor,
        "m7_controller": m7_controller,
        "m7_recovery_planner": m7_recovery_planner,
    }
