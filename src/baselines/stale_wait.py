from src.environment.actions import Action


class ReachableSetWaitBaseline:
    def __init__(self, threshold: int):
        if threshold < 1:
            raise ValueError("threshold must be >= 1")

        self.threshold = threshold

    def select_action(
        self,
        normal_action,
        reachable_size: int,
    ):
        if reachable_size >= self.threshold:
            return Action.WAIT

        return Action(normal_action)
