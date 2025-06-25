# https://www.kaggle.com/code/siddhantoon/ribonanzanet-2-0-ddpm-explained#Steps:
from typing import Optional

from torch import nn
import torch.nn.functional as F
import torch

from model.utils import default, exists


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention module, computing attention scores based on query and key similarity.

    :param temperature: Scaling factor for the dot product attention scores.
    :param attn_dropout: Dropout rate applied to attention weights.
    """

    def __init__(self, temperature: float, attn_dropout: float = 0.0):
        super().__init__()
        self.temperature: float = temperature
        self.dropout: nn.Dropout = nn.Dropout(attn_dropout) if attn_dropout > 0.0 else nn.Identity()

    def forward(
            self,
            q: torch.Tensor,  # (B, $, L_q, d_k) or (B, $, T, C) T is time, C is num_channels.
            k: torch.Tensor,  # (B, $, L_k, d_k)
            v: torch.Tensor,  # (B, $, L_k, d_v)
            attention_bias: Optional[torch.Tensor] = None,  # (B, $$, L_q, L_k) or None, $$ broadcastable to $ ($$ is 1s)
            attn_mask: Optional[torch.Tensor] = None  # (B, $$, L_q, L_k) or None,  ** broadcastable to $ ($$ is 1s)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Scaled Dot-Product Attention.

        :param q: Query tensor of shape (B, $, L_q, d_k), where B is batch size,
                  L_q is the query sequence length, and d_k is the key/query dimension.
        :param k: Key tensor of shape (B, $, L_k, d_k). L_k is the key sequence length.
        :param v: Value tensor of shape (B, $, L_k, d_v), where d_v is the value dimension.
        :param attention_bias: Optional bias mask tensor of shape (B, $$, L_q, L_k). $$ broadcastable to $ ($$ is 1s)
        :param attn_mask: Optional attention bool mask tensor of shape (B, $$, L_q, L_k), where False values indicate positions
                  to mask. Used for causal masking or padding. $$ broadcastable to $ ($$ is 1s)
        :return:
            output (torch.Tensor): The result of the attention mechanism, shape (B, $, L_q, d_v).
            attn (torch.Tensor): Attention weights after softmax and dropout, shape (B, $, L_q, L_k).
        """

        attn = torch.matmul(q, k.transpose(-2, -1)) / self.temperature  # Shape: (B, $, L_q, L_k)

        if attention_bias is not None:
            attn = attn + attention_bias  # Apply bias mask, Shape: (B, $, L_q, L_k)

        if attn_mask is not None:
            attn = attn.float().masked_fill(~attn_mask, float('-inf'))  # Apply attention mask, Shape: (B, $, L_q, L_k)

        attn = self.dropout(F.softmax(attn, dim=-1))  # Shape: (B, $, L_q, L_k)
        output = torch.matmul(attn, v)  # Shape: (B, $, L_q, d_v)

        return output, attn


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module
    :param d_model: The number of input features. or C num_channels in input.
    :param n_head: The number of heads to use.
    :param d_k: The dimensionality of the keys.
    :param d_v: The dimensionality of the values. If None, defaults to d_k.
    :param d_k_model: The dimensionality of the keys in the model. If None, defaults to d_model.
    :param d_v_model: The dimensionality of the values in the model. If None, defaults to d_k_model.
    :param dropout: The dropout rate to apply to the attention weights.
    """

    def __init__(
            self,
            d_model: int,
            n_head: int,
            d_k: int,
            d_v: Optional[int] = None,
            d_k_model: Optional[int] = None,  # If not provided, defaults to d_model.
            d_v_model: Optional[int] = None,  # If not provided, defaults to d_k_model.
            dropout: float = 0.1
    ):
        super().__init__()

        self.n_head: int = n_head
        self.d_k: int = d_k
        self.d_v: int = default(d_v, d_k)

        d_k_model = default(d_k_model, d_model)
        d_v_model = default(d_v_model, d_k_model)

        self.w_qs = nn.Linear(d_model, n_head * self.d_k, bias=False)  # (d_model) -> (n_head * d_k)
        self.w_ks = nn.Linear(d_k_model, n_head * self.d_k, bias=False)  # (d_k_model) -> (n_head * d_k)
        self.w_vs = nn.Linear(d_v_model, n_head * self.d_v, bias=False)  # (d_v_model) -> (n_head * d_v)
        self.fc = nn.Linear(n_head * self.d_v, d_model, bias=False)  # (n_head * d_v) -> (d_model)

        self.attention: ScaledDotProductAttention = ScaledDotProductAttention(
            temperature=self.d_k ** 0.5,
            attn_dropout=dropout
        )

    def forward(
            self,
            q: torch.Tensor,  # Shape: (B, $, L_q, d_model) # B, T, C
            k: torch.Tensor,  # Shape: (B, $, L_k, d_k_model)
            v: torch.Tensor,  # Shape: (B, $, L_k, d_v_model)
            attention_bias: Optional[torch.Tensor] = None,  # Shape: (B, $$, 1, L_q, L_k) or None, $$ broadcastable to $
            has_attention_bias_dim_for_heads: bool = False,  # If True, attention_bias is expected to be of shape (B, $$, x, L_q, L_k) where x in {1, n_head}, otherwise (B, $$, L_q, L_k).
            attn_mask: Optional[torch.Tensor] = None,  # Optional attention mask for causal masking or padding.
            is_attn_mask_1d: bool = True,  # If True, attn_mask is expected to be of shape (B, $$, L_q) and also L_q = L_k, otherwise (B, $$, L_q, L_k), $$ # broadcastable to $

    ) -> tuple[torch.Tensor, torch.Tensor]:  # Returns (output, attention)
        """
        Forward pass of the Multi-Head Attention module.
        :param q: Query tensor of shape (B, $, L_q, d_model), where B is batch size,
                    L_q is the query sequence length, and d_model is the model dimension.
        :param k: Key tensor of shape (B, $, L_k, d_model), where L_k is the key sequence length.
        :param v: Value tensor of shape (B, $, L_k, d_model)
        :param attention_bias: Optional bias tensor of shape (B, $$, L_q, L_k), (B, $$, 1, L_q, L_k), (B, $$, n_head, L_q, L_k) or None, where $$ is broadcastable to $.
        :param has_attention_bias_dim_for_heads: If True, attention_bias is expected to be of shape (B, $$, x, L_q, L_k) where x in {1, n_head},
                    otherwise (B, $$, L_q, L_k). This is used to handle cases where the bias is shared across heads.
        :param attn_mask: Optional attention mask tensor of shape (B, $$, L_q) or (B, $$, L_q, L_k), where $$ is broadcastable to $.
        :param is_attn_mask_1d: If True, attn_mask is expected to be of shape (B, $$, L_q) and also L_q = L_k,
                    otherwise (B, $$, L_q, L_k), where $$ is broadcastable to $.
        :return:
            output: The result of the attention mechanism, shape (B, $, L_q, d_model).
            attn: The attention weights after softmax and dropout, shape (B, $, n_head, L_q, L_k).
        """
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        B_, L_q, L_k = q.shape[:-2], q.size(-2), v.size(-2)  # B_ is (B, $).

        # Linear projections and reshape to multiple heads
        q = self.w_qs(q).view(*B_, L_q, n_head, d_k)  # Shape: (B, $, L_q, n_head, d_k)
        k = self.w_ks(k).view(*B_, L_k, n_head, d_k)  # Shape: (B, $, L_k, n_head, d_k)
        v = self.w_vs(v).view(*B_, L_k, n_head, d_v)  # Shape: (B, $, L_k, n_head, d_v)

        # Transpose for multi-head attention computation
        q, k, v = (
            q.transpose(-3, -2),  # Shape: (B, $, n_head, L_q, d_k)
            k.transpose(-3, -2),  # Shape: (B, $, n_head, L_k, d_k)
            v.transpose(-3, -2),  # Shape: (B, $, n_head, L_k, d_v)
        )

        if exists(attention_bias) and not has_attention_bias_dim_for_heads:
            attention_bias = attention_bias.clone().unsqueeze(-3)  # Shape: (B, $$, 1, L_q, L_k)

        if exists(attn_mask):
            if is_attn_mask_1d:  # If attn_mask is (B, $$, L_q) & L_q = L_k
                attn_mask = attn_mask.clone().unsqueeze(-1).short()  # Shape: (B, $$, L_q, 1)
                attn_mask = (torch
                    .matmul(
                        attn_mask,  # Shape: (B, $$, L_q, 1)
                        attn_mask.transpose(-2, -1),  # Shape: (B, $$, 1, L_q)
                    )  # Shape: (B, $$, L_q, L_q)
                    .unsqueeze(-3)  # Shape: (B, $$, 1, L_q, L_q)
                    .bool()
                )
            else:  # If attn_mask is (B, $$, L_q, L_k)
                attn_mask = attn_mask.clone().unsqueeze(-3)  # Shape: (B, $$, 1, L_q, L_k)

        # output: Shape: (B, $, n_head, L_q, d_v)
        # attn: Shape: (B, $, n_head, L_q, L_k)
        output, attn = self.attention(q, k, v, attention_bias=attention_bias, attn_mask=attn_mask)

        # Reshape back to original format
        output = output.transpose(-3, -2).contiguous().view(*B_, L_q, -1)  # Shape: (B, $, L_q, n_head * d_v)

        output = self.fc(output)  # Shape: (B, $, L_q, d_model)

        return output, attn
