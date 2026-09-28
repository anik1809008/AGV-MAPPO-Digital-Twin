from src.baselines.stale_wait import ReachableSetWaitBaseline
from src.safety.m4_controller import M4Controller
from src.safety.shield import SafetyShield
from src.safety.progress_monitor import ProgressMonitor

def build_method_controllers(
    method,
    actor,
    grid,
    m5_threshold=2,
):
    m4_controller = None
    m5_baseline = None
    m7_progress_monitor = None
    if method in {"M4", "M7"}:
        shield = SafetyShield(grid)

        m4_controller = M4Controller(
            actor=actor,
            shield=shield,
        )
    if method == "M7":
         m7_progress_monitor = ProgressMonitor(
            grid=grid,
            stagnation_steps=10,
         )
    if method == "M5":
        m5_baseline = ReachableSetWaitBaseline(
            threshold=m5_threshold,
        )


    return {
        "m4_controller": m4_controller,
        "m5_baseline": m5_baseline,
        "m7_progress_monitor": m7_progress_monitor,
    }
