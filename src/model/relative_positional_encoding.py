# https://www.kaggle.com/code/siddhantoon/ribonanzanet-2-0-ddpm-explained#Steps:
from torch import nn
import torch


class RelativePositionalEncoding(nn.Module):
    """
    Implements relative positional encoding for sequence-based models.
    :param dim: (int) The output embedding dimension.
    :param clip_value: (int) The maximum absolute value for relative positions.
        This value is used to clip the relative position indices to a manageable range.
        Default is 16, which means relative positions will be in the range [-16, 16].
    """

    def __init__(self, dim: int, clip_value: int = 16):
        super().__init__()
        self.clip_value: int = clip_value
        self.linear = nn.Linear(2 * self.clip_value + 1, dim)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Computes the relative positional encodings for a given sequence.

        :param src: Input tensor of shape (B, $, L, D), where:
            - B: Batch size
            - L: Sequence length
            - D: Feature dimension (ignored in this module)
        :return: Relative positional encoding of shape (L, L, dim)
        """
        L = src.shape[-2]  # Sequence length
        device = src.device
        res_id = torch.arange(L, device=device)  # Shape: (L,)

        bin_values = torch.arange(-self.clip_value, self.clip_value + 1, device=device)  # Shape: (2 * clip_value + 1,)

        d = res_id[:, None] - res_id[None, :]  # d[i, j] = i - j, Shape: (L, L)
        clip_value = torch.tensor(self.clip_value, device=device)

        # Clipping the values within the range [-clip_value, clip_value]
        d = d.clip(min=-clip_value, max=clip_value)  # Shape: (L, L)

        # One-hot encoding of relative positions
        d_onehot = (d[..., None] == bin_values).float()  # (L, L, 2 * clip_value + 1)

        assert d_onehot.sum(dim=-1).min() == 1, "Not proper one-hot encoding."

        # Linear transformation to embedding space
        p = self.linear(d_onehot)  # (L, L, 2 * clip_value + 1) -> (1, L, L, dim)

        return p  # (L, L, dim)
