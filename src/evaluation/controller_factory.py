from src.baselines.stale_wait import ReachableSetWaitBaseline
from src.safety.m4_controller import M4Controller
from src.safety.shield import SafetyShield


def build_method_controllers(
    method,
    actor,
    grid,
    m5_threshold=2,
):
    m4_controller = None
    m5_baseline = None

    if method == "M4":
        shield = SafetyShield(grid)

        m4_controller = M4Controller(
            actor=actor,
            shield=shield,
        )

    if method == "M5":
        m5_baseline = ReachableSetWaitBaseline(
            threshold=m5_threshold,
        )

    return {
        "m4_controller": m4_controller,
        "m5_baseline": m5_baseline,
    }
