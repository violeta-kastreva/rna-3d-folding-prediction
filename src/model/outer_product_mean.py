# https://www.kaggle.com/code/siddhantoon/ribonanzanet-2-0-ddpm-explained#Steps:
from typing import Optional

from einops import rearrange
from torch import nn
import torch


class OuterProductMean(nn.Module):
    """
    Outer Product Mean class.
    :param in_dim: Dimensionality of the input sequence representations.
    :param dim_msa: Intermediate lower-dimensional representation.
    :param pairwise_dim: Final dimensionality of the pairwise output.
    """
    def __init__(
        self,
        in_dim: int,
        dim_msa: int,
        pairwise_dim: int,
    ):
        super().__init__()
        self.proj_down1 = nn.Linear(in_dim, dim_msa)  # projects the input sequence representation into a lower dimensional space
        self.proj_down2 = nn.Linear(dim_msa ** 2, pairwise_dim)  # projects the outer product representation (reshaped) to the final pairwise_dim.

    def forward(
        self,
        seq_rep: torch.Tensor,  # Shape: (B, $, L, in_dim)
        pair_rep_bias: Optional[torch.Tensor] = None  # Shape: (B, $, L, L, pairwise_dim)
    ) -> torch.Tensor:
        """
        Forward pass of the OuterProductMean module.
        :param seq_rep: The input sequence representation (e.g. MSA) tensor of shape (B, $, L, in_dim).
              L is the sequence length, B is the batch size, and in_dim is the input dimensionality.
        :param pair_rep_bias: Optional pairwise representation bias tensor of shape (B, $, L, L, pairwise_dim).
              if provided, it will be added to the computed outer product.
        :return:
            The pairwise outer product representation tensor of shape (B, $, L, L, pairwise_dim).
        """
        seq_rep = self.proj_down1(seq_rep)  # Shape: (B, $, L, dim_msa)
        outer_product = torch.einsum('...id,...jc -> ...ijcd', seq_rep, seq_rep)  # Shape: (B, $, L, L, dim_msa, dim_msa)
        outer_product = rearrange(outer_product, '... i j c d -> ... i j (c d)')  # flattens the last two dimensions, Shape: (B, $, L, L, dim_msa^2).
        outer_product = self.proj_down2(outer_product)  # Shape: (B, $, L, L, pairwise_dim)

        if pair_rep_bias is not None:
            outer_product = outer_product + pair_rep_bias

        return outer_product

