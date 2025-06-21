# https://www.kaggle.com/code/siddhantoon/ribonanzanet-2-0-ddpm-explained#Steps:
from typing import Optional, Literal

from torch import einsum
from torch import nn
import torch

from model.utils import default, exists


class TriangleMultiplicativeModule(nn.Module):
    """
    This class is applied to the pairwise residue representations.
    Inspired by the triangle inequality principle.
    Analogous to the GRUs, but for 2D per dimension.
    :param dim: Input feature dimension.
    :param hidden_dim: Hidden dimension for the intermediate representation. If None, defaults to dim.
    :param mix: Specifies the mixing strategy, either "ingoing" or "outgoing".
    """

    def __init__(
            self,
            *,
            dim: int,
            hidden_dim: Optional[int] = None,
            mix: Literal["ingoing", "outgoing"] = "ingoing",
    ):
        super().__init__()
        assert mix in {"ingoing", "outgoing"}, 'mix must be either "ingoing" or "outgoing"'

        hidden_dim: int = default(hidden_dim, dim)
        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim)
        self.right_proj = nn.Linear(dim, hidden_dim)
        self.left_gate = nn.Linear(dim, hidden_dim)
        self.right_gate = nn.Linear(dim, hidden_dim)
        self.out_gate = nn.Linear(dim, dim)

        # Initialize all gating to identity
        for gate in (self.left_gate, self.right_gate, self.out_gate):
            nn.init.constant_(gate.weight, 0.)
            nn.init.constant_(gate.bias, 1.)

        if mix == 'outgoing':
            self.mix_einsum_eq = '... i k d, ... j k d -> ... i j d'
        elif mix == 'ingoing':
            self.mix_einsum_eq = '... k j d, ... k i d -> ... i j d'

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim)

    def forward(
            self,
            x: torch.Tensor,  # Shape: (B, $, L, L, dim)
            sequence_mask: Optional[torch.Tensor] = None  # Shape: (B, $$, L), $$ broadcastable to $ ($$ is 1s)
    ) -> torch.Tensor:  # Shape: (B, $, L, L, dim)
        """
        Forward pass of the TriangleMultiplicativeModule.
        :param x: The input tensor of shape (B, $, L, L, dim), where B is the batch size,
        $ is any other type of batching (could be nothing), L is the sequence length, and dim is the feature dimension.
        This tensor represents the pairwise residue representations.
        :param sequence_mask: An optional causal/padding sequence mask tensor of shape (B, $$, L), $$ is broadcastable to $.
        :return: The output tensor of the same shape as x, where the pairwise residue representations have been modified
        """
        assert x.shape[-3] == x.shape[-2], 'Feature map must be symmetrical.'

        x = self.norm(x)  # Shape: (B, $, L, L, dim)

        left = self.left_proj(x)  # Shape: (B, $, L, L, hidden_dim)
        right = self.right_proj(x)  # Shape: (B, $, L, L, hidden_dim)

        if exists(sequence_mask):
            sequence_mask = sequence_mask.unsqueeze(-1).float()  # Shape: (B, $$, L, 1)
            mask = torch.matmul(
                sequence_mask,  # Shape: (B, $$, L, 1)
                sequence_mask.transpose(-2, -1),  # Shape: (B, $$, 1, L)
            )  # Shape: (B, $$, L, L)
            mask = mask.unsqueeze(-1)  # (B, $$, L, L, 1)

            left = left * mask  # Shape: (B, $, L, L, hidden_dim)
            right = right * mask  # Shape: (B, $, L, L, hidden_dim)

        left_gate = self.left_gate(x).sigmoid()  # Shape: (B, $, L, L, hidden_dim)
        right_gate = self.right_gate(x).sigmoid()  # Shape: (B, $, L, L, hidden_dim)
        out_gate = self.out_gate(x).sigmoid()  # Shape: (B, $, L, L, dim)

        left = left * left_gate  # Shape: (B, $, L, L, hidden_dim)
        right = right * right_gate  # Shape: (B, $, L, L, hidden_dim)

        # mix == 'outgoing': mix_einsum_eq = '... i k d, ... j k d -> ... i j d'
        # mix == 'ingoing': mix_einsum_eq = '... k j d, ... k i d -> ... i j d'
        out = einsum(self.mix_einsum_eq, left, right)  # Shape: (B, $, L, L, hidden_dim)

        out = self.to_out_norm(out)  # Shape: (B, $, L, L, hidden_dim)
        out = self.to_out(out)  # Shape: (B, $, L, L, dim)
        out = out * out_gate  # Shape: (B, $, L, L, dim)
        return out
