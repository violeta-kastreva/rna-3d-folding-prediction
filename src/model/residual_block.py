import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, *layers: nn.Module):
        super().__init__()
        self.layers: nn.ModuleList = nn.ModuleList(layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        for layer in self.layers:
            x = layer(x)

        return input + x
