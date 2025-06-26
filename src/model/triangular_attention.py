# https://www.kaggle.com/code/siddhantoon/ribonanzanet-2-0-ddpm-explained#Steps:
from typing import Optional, Literal

from einops import rearrange
from torch import nn
import torch

from model.utils import default


class TriangularAttention(nn.Module):
    """
    Implements Triangular Attention Mechanism.
    Inspired by the triangle inequality principle.
    Analogous to the Self-Attention, but for 2D per dimension.
    :param d_model: Input feature dimension.
    :param d_k: Dimension of the key/query vectors per head.
    :param d_v: Dimension of the value vectors per head. If None, defaults to d_k.
    :param n_heads: Number of attention heads.
    :param wise: Whether to apply row-wise or column-wise attention.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_k: int,
        d_v: Optional[int] = None,
        wise: Literal["row", "col"] = "row",
    ):
        super().__init__()
        assert wise in {"row", "col"}, '"wise" must be either "row" or "col".'

        self.n_heads: int = n_heads
        self.wise = wise
        self.d_k: int = d_k
        self.d_v: int = default(d_v, d_k)
        self.scale: float = self.d_k ** 0.5

        self.norm = nn.LayerNorm(d_model)

        self.q_proj = nn.Linear(d_model, n_heads * self.d_k, bias=False)  # (d_model) -> (n_head * d_k)
        self.k_proj = nn.Linear(d_model, n_heads * self.d_k, bias=False)  # (d_model) -> (n_head * d_k)
        self.v_proj = nn.Linear(d_model, n_heads * self.d_v, bias=False)  # (d_model) -> (n_head * d_v)

        self.w_bias = nn.Linear(d_model, n_heads, bias=False)

        self.out_gate = nn.Sequential(
            nn.Linear(d_model, n_heads * self.d_v),
            nn.Sigmoid(),
        )

        self.out_proj = nn.Linear(n_heads * self.d_v, d_model)

    def forward(
            self,
            z: torch.Tensor,  # Shape: (B, $, I, J, d_model)
            sequence_mask: torch.Tensor  # Shape: (B, $$, I, J), $$ broadcastable to $
    ) -> torch.Tensor:  # Shape: (B, $, I, J, d_model)
        """
        Forward pass for TriangularAttention.
        :param z: Input tensor of shape (B, $, I, J, d_model). I,J are actually L but here we call them I & J.
                It's actually (B, $, L, L, d_model).
        :param sequence_mask: Causal/Padding sequence mask of shape (B, $$, I). $$ broadcastable to $.
        :return: Output tensor of shape (B, $, I, J, d_model).

        [i] comments are from: https://kaggle-images.s3.us-west-2.amazonaws.com/ribonanza-3d/triangular_self_attention.png
        """
        # Spawn pair mask
        sequence_mask = sequence_mask.clone().float()  # Shape: (B, $$, I)
        sequence_mask = sequence_mask.unsqueeze(-1)  # Shape: (B, $$, I, 1)
        attn_mask = torch.matmul(
            sequence_mask,  # Shape: (B, $$, I, 1)
            sequence_mask.transpose(-2, -1),  # Shape: (B, $$, I, 1) # I = J
        ).bool()  # Shape: (B, $$, I, J)

        wise = self.wise
        z = self.norm(z)  # (B, $, I, J, d_model)

        # Compute Q, K, V
        q = self.q_proj(z)  # [4], Shape: (B, $, I, J, n_heads * d_k)
        k = self.k_proj(z)  # [3], Shape: (B, $, I, J, n_heads * d_k)
        v = self.v_proj(z)  # [2], Shape: (B, $, I, J, n_heads * d_v)

        # Unflatten last dimension to separate heads
        q = self._unflatten_last_dim(q)  # [4], Shape: (B, $, I, J, n_heads, d_k)
        k = self._unflatten_last_dim(k)  # [3], Shape: (B, $, I, J, n_heads, d_k)
        v = self._unflatten_last_dim(v)  # [2], Shape: (B, $, I, J, n_heads, d_v)

        # Compute attention bias
        attention_bias = self.w_bias(z)  # [5], Shape: (B, $, I, J, n_heads)

        if wise == "row":
            attention_eq: str = "...rihd,...rjhd->...rijh"
            weighted_avg_of_values_eq: str = "...rijh,...rjhd->...rihd"
            attention_bias = rearrange(attention_bias, "... i j (r h) -> ... r i j h", r=1)  # Shape: (B, $, 1, I, J, n_heads)
            softmax_dim: int = -2
            attn_mask = rearrange(attn_mask, "... i j -> ... 1 i j 1")  # Shape: (B, $$, 1, I, J, 1)
        elif wise == "col":
            attention_eq: str = "...ilhd,...jlhd->...ijlh"
            weighted_avg_of_values_eq: str = "...ijlh,...jlhd->...ilhd"
            attention_bias = rearrange(attention_bias, "... i j (l h) -> ... i j l h", l=1)  # Shape: (B, $$, I, J, 1, n_heads)
            softmax_dim: int = -3
            attn_mask = rearrange(attn_mask, "... i j -> ... i j 1 1")  # Shape: (B, I, J, 1, 1)
        else:
            raise ValueError("'wise' should be 'col' or 'row'.")

        # For [6], [7], [8] & [9]:
        #   q: Shape: (B, $, I, J, n_heads, d_k)
        #   k: Shape: (B, $, I, J, n_heads, d_k)
        #   v: Shape: (B, $, I, J, n_heads, d_v)
        #   row:
        #       attention_eq: "...rihd,...rjhd->...rijh"
        #       [6]: einsum(q, k): Shape: (B, $, I, J, J, n_heads)
        #       attention_bias:    Shape: (B, $, 1, I, J, n_heads)  # I = J
        #       [7]: attn_logits:  Shape: (B, $, I, J, J, n_heads)
        #       attn_mask:         Shape: (B, $$, 1, I, J, 1)  # I = J
        #       [8]: attn:         Shape: (B, $, I, J, J, n_heads)
        #                                              ^ softmax_dim = -2
        #       weighted_avg_of_values_eq: "...rijh,...rjhd->...rihd"
        #       [9]: out:          Shape: (B, $, I, J, n_heads, d_v)
        #   col:
        #       attention_eq: "...ilhd,...jlhd->...ijlh"
        #       [6]: einsum(q, k): Shape: (B, $, I, I, J, n_heads)
        #       attention_bias:    Shape: (B, $, I, J, 1, n_heads)  # J = I
        #       [7]: attn_logits:  Shape: (B, $, I, I, J, n_heads)
        #       attn_mask:         Shape: (B, $$, I, J, 1, 1)  # J = I
        #       [8]: attn:         Shape: (B, $, I, I, J, n_heads)
        #                                           ^ softmax_dim = -3
        #       weighted_avg_of_values_eq: "...ijlh,...jlhd->...ilhd"
        #       [9]: out:          Shape: (B, $, I, J, n_heads, d_v)

        # Compute attention logits
        attention_unbiased_logits = torch.einsum(attention_eq, q, k) / self.scale  # [6], Shape: row: (B, $, I, J, J, n_heads), col: (B, $, I, I, J, n_heads)
        attn_logits = attention_unbiased_logits + attention_bias  # [7], Shape: row: (B, $, I, J, J, n_heads), col: (B, $, I, I, J, n_heads)
        attn_logits = attn_logits.masked_fill(~attn_mask, float("-inf"))  # Apply causal mask. The shape remains unchanged.

        # Compute attention weights
        attn = attn_logits.softmax(dim=softmax_dim)  # [8], Shape: row: (B, I, J, J, n_heads), col: (B, I, I, J, n_heads)

        # Compute attention output
        output = torch.einsum(weighted_avg_of_values_eq, attn, v)  # [9], Shape: (B, $, I, J, n_heads, d_v)

        # Flatten last dimension to combine heads
        flattened_output = rearrange(output, "... i j h d -> ... i j (h d)")  # Shape: (B, $, I, J, n_heads * d_v)

        # Compute output gate & apply it to the flattened output
        out_gate = self.out_gate(z)  # [1], Shape: (B, $, I, J, n_heads * d_v)
        gated_output = out_gate * flattened_output  # [10], Shape: (B, I, J, n_heads * d_v)

        # Final projection
        projected_output = self.out_proj(gated_output)  # Shape: (B, I, J, d_model)
        return projected_output

    def _unflatten_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(x, "... (h d) -> ... h d", h=self.n_heads)
