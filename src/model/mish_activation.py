from torch import tanh
import torch.nn as nn

from torch.nn.functional import softplus


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x * tanh(softplus(x))
