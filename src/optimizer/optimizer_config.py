from dataclasses import dataclass


@dataclass
class OptimizerConfig:
    lr: float = 1e-3
    lr_scheduler_eta_min: float = 1e-6
    num_warmup_iterations: int = 100
