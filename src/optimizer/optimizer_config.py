from dataclasses import dataclass


@dataclass
class OptimizerConfig:
    lr: float = 0.001
    clip_gradient_value: float = 5.0