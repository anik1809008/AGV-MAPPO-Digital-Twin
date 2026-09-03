import random


class ExecutionDelayModel:
    def __init__(self, immediate_probability=0.8, seed=None):
        if not 0.0 <= immediate_probability <= 1.0:
            raise ValueError("immediate_probability must be between 0 and 1")

        self.immediate_probability = immediate_probability
        self.random = random.Random(seed)

    def sample_delay(self):
        if self.random.random() < self.immediate_probability:
            return 0

        return 1
