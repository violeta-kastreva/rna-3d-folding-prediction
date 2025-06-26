from typing import Any

import numpy as np
import torch
from torch import nn


class GradientAutoClipper:
    """
    A class to automatically clip gradients during training.
    This is useful to prevent exploding gradients in deep learning models.
    """

    def __init__(
        self,
        model: nn.Module,
        unclipped_warmup_steps: int,
        percentile: float,
    ):
        """
        :param percentile: The percentile of the gradient norm to clip to.
        """
        assert 0 < percentile < 1, "Percentile must be between 0 and 1."
        self.model: nn.Module = model
        self.unclipped_warmup_steps: int = unclipped_warmup_steps
        self.clip_percentile: int = int(100 * percentile)
        self.gradient_norm_history: list[float] = []

    def get_gradient_norm(self) -> torch.Tensor:
        return torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in self.model.parameters()]), 2)

    def step(self) -> None:
        observed_grad_norm: torch.Tensor = self.get_gradient_norm()
        self.gradient_norm_history.append(observed_grad_norm.item())

        if len(self.gradient_norm_history) < self.unclipped_warmup_steps:
            # During warmup, do not clip gradients
            return

        clip_value: float = float(np.percentile(self.gradient_norm_history, self.clip_percentile))
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)

    def state_dict(self) -> dict[str, Any]:
        return {
            "gradient_norm_history": self.gradient_norm_history,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if "gradient_norm_history" in state_dict:
            self.gradient_norm_history = state_dict["gradient_norm_history"]

