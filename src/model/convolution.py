from typing import Literal, Union

import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd, Conv1d, Conv2d, Conv3d, _size_2_t


class ConvNd(nn.Module):
    """
    A wrapper class for 1D, 2D, and 3D convolutional layers, that allows multiple batching dimensions.
    """
    _CONV_MAP: dict[Literal[1, 2, 3], _ConvNd] = {
        1: Conv1d,
        2: Conv2d,
        3: Conv3d,
    }
    
    def __init__(
        self,
        n: Literal[1, 2, 3],
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.n: Literal[1, 2, 3] = n
        self.conv: _ConvNd = self._CONV_MAP[n](
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the convolutional layer.
        :param x: Input tensor of shape (B, $, C, L) for Conv1d, (B, $, C, H, W) for Conv2d, or (B, $, C, D, H, W) for Conv3d.
        :return: The convoluted tensor with the C dimension replaced by out_channels.
        """
        flattened_x = x.view(-1, *x.shape[-self.n - 1:])
        output = self.conv(flattened_x)
        return output.view(*x.shape[:-self.n - 1], *output.shape[1:])
