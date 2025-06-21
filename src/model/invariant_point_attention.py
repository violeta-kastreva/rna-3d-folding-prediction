# https://github.com/lucidrains/invariant-point-attention
from typing import Optional
import math

import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch import nn, einsum
from einops.layers.torch import Rearrange
from einops import rearrange

from model.utils import default, exists, disable_tf32


class InvariantPointAttention(nn.Module):
    """
    Invariant Point Attention (IPA) module.
    Implements the invariant point attention mechanism as described in the AlphaFold2 paper.
    :param d_model: Input feature dimension.
    :param n_heads: Number of attention heads.
    :param d_sk: Dimension of the scalar key vectors per head.
    :param d_pk: Dimension of the point key vectors per head.
    :param d_sv: Dimension of the scalar value vectors per head. If None, defaults to d_sk.
    :param d_pv: Dimension of the point value vectors per head. If None, defaults to d_pk.
    :param d_pair_repr: Dimension of the pairwise representation. If None, defaults to dim.
    :param does_require_pair_repr: Whether the pairwise representation is required.
    :param eps: Small value to avoid division by zero in attention calculations.
    """

    DIMS: int = 3

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        d_sk: int,
        d_pk: int,
        d_sv: Optional[int] = None,
        d_pv: Optional[int] = None,
        d_pair_repr: Optional[int] = None,
        does_require_pair_repr: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()

        self.eps: float = eps
        self.n_heads: int = n_heads
        self.does_require_pair_repr: bool = does_require_pair_repr

        d_sv = default(d_sv, d_sk)
        d_pv = default(d_pv, d_pk)
        self.d_sk: int = d_sk
        self.d_sv: int = d_sv
        self.d_pk: int = d_pk
        self.d_pv: int = d_pv

        # Num attention contributions
        num_attn_logits: int = 3 if does_require_pair_repr else 2

        # QKV projection for scalar attention (normal)
        self.scalar_attn_logits_scale: float = (num_attn_logits * d_sk) ** -0.5

        self.qkv_scalar_proj = nn.Linear(d_model, n_heads * (d_sk + d_sk + d_sv), bias=False)

        # QKV projection for point attention (coordinate and orientation aware)
        self.point_weights = nn.Parameter(torch.full((n_heads,), math.log(math.e - 1)))

        self.point_attn_logits_scale = ((num_attn_logits * d_pk) * ((self.DIMS ** 2) / 2)) ** -0.5

        self.qkv_point_proj = nn.Linear(d_model, n_heads * (d_pk + d_pk + d_pv) * self.DIMS, bias=False)

        # Pairwise representation projection to attention bias
        if does_require_pair_repr:
            d_pair_repr = default(d_pair_repr, d_model)
            self.pairwise_attn_logits_scale = num_attn_logits ** -0.5

            self.to_pairwise_attn_bias = nn.Sequential(
                nn.Linear(d_pair_repr, n_heads),
                Rearrange("... i j h -> ... h i j"),
            )
        else:
            d_pair_repr = 0

        # Combined outputs: n_heads * (scalar dim + point dim * (DIMS for coords in R^DIMS + 1 for norm) + pairwise dim)
        self.out_proj = nn.Linear(n_heads * (d_sv + d_pv * (self.DIMS + 1) + d_pair_repr), d_model)

    def forward(
        self,
        single_repr: torch.Tensor,  # Shape: (B, $, L, d_model)
        pairwise_repr: Optional[torch.Tensor] = None,  # Shape: (B, $, L, L, d_pair_repr) if does_require_pair_repr else None
        *,
        rotations: torch.Tensor,  # Shape: (B, $$, L, DIMS, DIMS), $$ broadcastable to $
        translations: torch.Tensor,  # Shape: (B, $$, L, DIMS), $$ broadcastable to $
        sequence_mask: Optional[torch.Tensor] = None,  # If not None, Shape: (B, $$, L), $$ broadcastable to $
    ):
        """
        Forward pass of the Invariant Point Attention module.
        :param single_repr: Sequence representation tensor of shape (B, $, L, d_model), where B is the batch size,
            $ is any other type of batching (could be nothing), L is the sequence length, and d_model is the input
            feature dimension.
        :param pairwise_repr: Optional pairwise representation tensor of shape (B, $, L, L, d_pair_repr).
            If **does_require_pair_repr** is True, this parameter should be given.
        :param rotations: Rotation matrices tensor of shape (B, $$, L, 3, 3), where $$ is broadcastable to $.
        :param translations: Translation vectors tensor of shape (B, $$, L, 3), where $$ is broadcastable to $.
        :param sequence_mask: Optional causal/padding sequence mask tensor of shape (B, $$, L), where $$ is
            broadcastable to $.
        :return:
            Output update tensor for the input, of the same shape: (B, $, L, d_model).
        """
        x, n_heads, does_require_pair_repr = single_repr, self.n_heads, self.does_require_pair_repr
        d_sk, d_pk, d_sv, d_pv = self.d_sk, self.d_pk, self.d_sv, self.d_pv

        assert not (does_require_pair_repr and not exists(pairwise_repr)), \
            "Pairwise representation must be given as second argument."

        # Unsqueeze rotations and translations to match the expected shapes
        rotations = rotations[..., None, :, :, :]  # Shape: (B, $$, 1, L, DIMS, DIMS)
        translations = translations[..., None, :, None, :]  # Shape: (B, $$, 1, L, 1, DIMS)

        # Project queries, keys, values for scalar and point (coordinate-aware) attention pathways.
        # For points, the data is not split into Q, K, V, but rather QKV points are projected together for
        # matrix multiplication optimization (bigger matrices -> better operation parallelism).
        qkv_scalar = self.qkv_scalar_proj(x)  # Shape: (B, $, L, n_heads * (d_sk + d_sk + d_sv))
        (
            q_scalar,  # Shape: (B, $, L, n_heads * d_sk)
            k_scalar,  # Shape: (B, $, L, n_heads * d_sk)
            v_scalar,  # Shape: (B, $, L, n_heads * d_sv)
        ) = torch.split(
            qkv_scalar, [n_heads * d_sk, n_heads * d_sk, n_heads * d_sv], dim=-1,
        )

        qkv_point = self.qkv_point_proj(x)  # Shape: (B, $, L, n_heads * (d_sk + d_sk + d_sv) * DIMS)

        # Unflatten the last dimension to separate heads, point coordinates and representations
        qkv_point = self._unflatten_last_dim_point(qkv_point)  # Shape: (B, $, n_heads, L, d_pk + d_pk + d_pv, DIMS)

        # Transform QKV points into global frame with an affine angle-preserving transformation
        qkv_point = self.transform_from_local_to_global_frame(qkv_point, rotations, translations)  # Shape: (B, $, n_heads, L, d_pk + d_pk + d_pv, DIMS)

        (
            q_point,  # Shape: (B, $, n_heads, L, d_pk, DIMS)
            k_point,  # Shape: (B, $, n_heads, L, d_pk, DIMS)
            v_point,  # Shape: (B, $, n_heads, L, d_pv, DIMS)
        ) = torch.split(
            qkv_point,
            [d_pk, d_pk, d_pv],
            dim=-2,
        )

        q_point = q_point.contiguous()
        k_point = k_point.contiguous()
        v_point = v_point.contiguous()

        # Split out heads in scalar representations
        q_scalar = self._unflatten_last_dim_scalar(q_scalar)  # Shape: (B, $, n_heads, L, d_sk)
        k_scalar = self._unflatten_last_dim_scalar(k_scalar)  # Shape: (B, $, n_heads, L, d_sk)
        v_scalar = self._unflatten_last_dim_scalar(v_scalar)  # Shape: (B, $, n_heads, L, d_sv)

        # Derive attn logits for scalar and pairwise
        attn_logits_scalar = torch.matmul(q_scalar, k_scalar.transpose(-2, -1)) * self.scalar_attn_logits_scale  # Shape: (B, $, n_heads, L, L)

        attn_logits_pairwise: Optional[torch.Tensor] = None
        if does_require_pair_repr:
            attn_logits_pairwise = self.to_pairwise_attn_bias(pairwise_repr) * self.pairwise_attn_logits_scale  # Shape: (B, $, n_heads, L, L)

        # Derive attn logits for point attention
        # point_qk_diff[..., i, j, ...] = q_point[..., i, ...] - k_point[..., j, ...]
        point_qk_diff = (
            q_point[..., :, None, :, :]  # Shape: (B, $, n_heads, L, 1, d_pk, DIMS)
            - k_point[..., None, :, :, :]  # Shape: (B, $, n_heads, 1, L, d_pk, DIMS)
        )  # Shape: (B, $, n_heads, L, L, d_pk, DIMS)
        sum_point_dist_sq = (point_qk_diff ** 2).sum(dim=(-1, -2))  # Shape: (B, $, n_heads, L, L)

        point_weights: torch.Tensor = F.softplus(self.point_weights)  # Shape: (n_heads,)
        point_weights = point_weights[:, None, None]  # Shape: (n_heads, 1, 1)

        attn_logits_point = -0.5 * sum_point_dist_sq * point_weights * self.point_attn_logits_scale  # Shape: (B, $, n_heads, L, L)

        # Combine attn logits
        attn_logits = attn_logits_scalar + attn_logits_point  # Shape: (B, $, n_heads, L, L)

        if does_require_pair_repr:
            attn_logits = attn_logits + attn_logits_pairwise  # Shape: (B, $, n_heads, L, L)

        # Apply causal/padding mask
        if exists(sequence_mask):
            sequence_mask = sequence_mask[..., :, None] * sequence_mask[..., None, :]  # Shape: (B, $$, L, L)
            sequence_mask = sequence_mask[..., None, :, :]  # Shape: (B, $$, 1, L, L)
            attn_logits = attn_logits.masked_fill(~sequence_mask, float("-inf"))  # Shape: (B, $, n_heads, L, L)

        # Compute attention weights
        attn = attn_logits.softmax(dim=-1)  # Shape: (B, $, n_heads, L, L)

        with disable_tf32(), autocast("cuda", enabled=False):
            # Disable TF32 for precision

            # Aggregate scalar values
            results_scalar = torch.matmul(attn, v_scalar)  # Shape: (B, $, n_heads, L, d_sv)

            # Aggregate pairwise representation if given
            if does_require_pair_repr:
                results_pairwise = einsum(
                    "... h i j, ... i j d -> ... h i d",
                    attn,  # Shape: (B, $, n_heads, L, L)
                    pairwise_repr,  # Shape: (B, $, L, L, d_pair_repr)
                )  # Shape: (B, $, n_heads, L, d_pair_repr)

            # Aggregate point values
            results_points = einsum(
                "... i j, ... j d c -> ... i d c",
                attn,  # Shape: (B, $, n_heads, L, L)
                v_point,  # Shape: (B, $, n_heads, L, d_pv, DIMS)
            )  # Shape: (B, $, n_heads, L, d_pv, DIMS)

            # Rotate aggregated point values back into local frame
            results_points = self.transform_from_global_to_local_frame(
                results_points, rotations, translations,
            )  # Shape: (B, $, n_heads, L, d_pv, DIMS)

            results_points_norm = torch.sqrt(torch.square(results_points).sum(dim=-1) + self.eps)  # Shape: (B, $, n_heads, L, d_pv)

        # Merge back heads in the last dimension
        results_scalar = rearrange(
            results_scalar,  # Shape: (B, $, n_heads, L, d_sv)
            "... h l d -> ... l (h d)",
        )  # Shape: (B, $, L, n_heads * d_sv)
        results_points = rearrange(
            results_points,  # Shape: (B, $, n_heads, L, d_pv, DIMS)
            "... h l d c -> ... l (h d c)",
        )  # Shape: (B, $, L, n_heads * d_pv * DIMS)
        results_points_norm = rearrange(
            results_points_norm,  # Shape: (B, $, n_heads, L, d_pv)
            "... h l d -> ... l (h d)",
        )  # Shape: (B, $, L, n_heads * d_pv)

        results = (results_scalar, results_points, results_points_norm)

        if does_require_pair_repr:
            results_pairwise = rearrange(
                results_pairwise,  # Shape: (B, $, n_heads, L, d_pair_repr)
                "... h l d -> ... l (h d)",
            )  # Shape: (B, $, L, n_heads * d_pair_repr)
            results = (*results, results_pairwise)

        # Concatenate results $ project out
        results = torch.cat(results, dim=-1)  # Shape: (B, $, L, n_heads * (d_sv + d_pv * DIMS + d_pair_repr))

        output = self.out_proj(results)  # Shape: (B, $, L, d_model)
        return output

    def _unflatten_last_dim_scalar(self, t: torch.Tensor) -> torch.Tensor:
        """
        Unflattens the last dimension of a tensor to separate heads.
        :param t: Tensor of shape (..., L, n_heads * d).
        :return: Tensor of shape (..., n_heads, L, d).
        """
        return rearrange(t, "... l (h d) -> ... h l d", h=self.n_heads)

    def _unflatten_last_dim_point(self, t: torch.Tensor) -> torch.Tensor:
        """
        Unflattens the last dimension of a tensor to separate heads and dimensions.
        :param t: Tensor of shape (..., L, n_heads * d * DIMS).
        :return: Tensor of shape (..., n_heads, L, d, DIMS).
        """
        return rearrange(t, "... l (h d c) -> ... h l d c", h=self.n_heads, c=self.DIMS)

    @staticmethod
    def transform_from_local_to_global_frame(
        points: torch.Tensor,  # Shape: (B, $, d, DIMS)
        rotations: torch.Tensor,  # Shape: (B, $$, DIMS, DIMS)
        translations: torch.Tensor,  # Shape: (B, $$, 1, DIMS)
    ) -> torch.Tensor:  # Shape: (B, $, d, DIMS)
        """
        Rotate QKV points into global frame with an affine angle-preserving transformation.
        This is the inverse operation of **transform_from_global_to_local_frame**.

        :param points: Points coordinates tensor in the local frame of shape (B, $, d, DIMS).
        :param rotations: Rotation matrices of shape (B, $$, DIMS, DIMS); $$ is broadcastable to $.
        :param translations: Translation vectors of shape (B, $$, 1, DIMS).
        :return: Transformed "points" coordinates tensor in the global frame of the same shape.
        """
        return (
            torch.matmul(
                points,  # Shape: (B, $, d, DIMS)
                rotations,  # Shape: (B, $$, DIMS, DIMS)
            )  # Shape: (B, $, d, DIMS)
            + translations  # Shape: (B, $$, 1, DIMS)
        )  # Shape: (B, $, d, DIMS)

    @staticmethod
    def transform_from_global_to_local_frame(
        points: torch.Tensor,  # Shape: (B, $, d, DIMS)
        rotations: torch.Tensor,  # Shape: (B, $$, DIMS, DIMS)
        translations: torch.Tensor,  # Shape: (B, $$, 1, DIMS)
    ) -> torch.Tensor:  # Shape: (B, $, d, DIMS)
        """
        Rotate QKV points into local frame with an affine angle-preserving transformation.
        This is the inverse operation of  **transform_from_local_to_global_frame**.

        :param points: Points coordinates tensor in the global frame of shape (B, $, d, DIMS).
        :param rotations: Rotation matrices of shape (B, $$, DIMS, DIMS); $$ is broadcastable to $.
        :param translations: Translation vectors of shape (B, $$, 1, DIMS).
        :return: Transformed "points" coordinates tensor in the local frame of the same shape.
        """
        # Rotation matrices are orthogonal, so we can use transpose them to get their inverses.
        return torch.matmul(
                points  # Shape: (B, $, d, DIMS)
                - translations,  # Shape: (B, $$, 1, DIMS)
                rotations.transpose(-2, -1),  # Shape: (B, $$ DIMS, DIMS)
            )  # Shape: (B, $, d, DIMS)
