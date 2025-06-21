from typing import Union
from torch.nn.parameter import Parameter
from torch import nn
import torch.nn.functional as F
import torch


class GeMPooling(nn.Module):
    def __init__(self, p: float = 3, eps: float = 1e-6):
        super().__init__()
        self.p: torch.Tensor = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gem_pool(x, p=self.p, eps=self.eps)

    @classmethod
    def gem_pool(cls, x: torch.Tensor, p: Union[float, torch.Tensor], eps: float) -> torch.Tensor:
        return F.avg_pool1d(x.clamp(min=eps).pow(p), (x.size(-1))).pow(1. / p)

    def __repr__(self) -> str:
        return (
            self.__class__.__name__ +
            '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' +
            'eps=' + str(self.eps) + ')'
        )
