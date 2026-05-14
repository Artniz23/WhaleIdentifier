import math

class WarmupCosineLambda:
    def __init__(self, warmup_steps: int, cycle_steps: int, decay_scale: float, exponential_warmup: bool = False):
        self.warmup_steps = warmup_steps
        self.cycle_steps = cycle_steps
        self.decay_scale = decay_scale
        self.exponential_warmup = exponential_warmup

    def __call__(self, epoch: int):
        if epoch < self.warmup_steps:
            if self.exponential_warmup:
                return self.decay_scale * pow(self.decay_scale, -epoch / self.warmup_steps)
            ratio = epoch / self.warmup_steps
        else:
            ratio = (1 + math.cos(math.pi * (epoch - self.warmup_steps) / self.cycle_steps)) / 2
        return self.decay_scale + (1 - self.decay_scale) * ratio