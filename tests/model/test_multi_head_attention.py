import pytest
import torch

from model.multi_head_attention import MultiHeadAttention


@pytest.mark.parametrize(
    "B, d_model, n_head, d_k, d_v, L_q, L_k, has_attention_bias_dim_for_heads, is_attn_mask_1d, dollar",
    [
        (2, 16, 4, 8, 8, 10, 12, has_attention_bias_dim_for_heads, is_attn_mask_1d, dollar)
        for has_attention_bias_dim_for_heads in (True, False)
        for is_attn_mask_1d in (True, False)
        for dollar in (tuple(), (3, 2))
    ]
)
def test_multi_head_attention_forward(
        B, d_model, n_head, d_k, d_v, L_q, L_k, has_attention_bias_dim_for_heads, is_attn_mask_1d, dollar
):
    if is_attn_mask_1d:
        L_q = L_k  # If is_attn_mask_1d, L_q must equal L_k

    mha = MultiHeadAttention(
        d_model=d_model,
        n_head=n_head,
        d_k=d_k,
        d_v=d_v,
        dropout=0.1,
    )

    q = torch.randn(B, *dollar, L_q, d_model)
    k = torch.randn(B, *dollar, L_k, d_model)
    v = torch.randn(B, *dollar, L_k, d_model)

    if has_attention_bias_dim_for_heads:
        attention_bias = torch.randn(B, *(1 for _ in dollar), n_head, L_q, L_k)
    else:
        attention_bias = torch.randn(B, *(1 for _ in dollar), L_q, L_k)  # Optional bias tensor

    if is_attn_mask_1d:
        attn_mask = torch.ones(B, *(1 for _ in dollar), L_q, dtype=torch.bool)
    else:
        attn_mask = torch.ones(B, *(1 for _ in dollar), L_q, L_k,
                                dtype=torch.bool)  # Optional attention mask

    output, attn = mha(q, k, v,
                       attention_bias=attention_bias,
                       has_attention_bias_dim_for_heads=has_attention_bias_dim_for_heads,
                       attn_mask=attn_mask,
                       is_attn_mask_1d=is_attn_mask_1d)

    assert output.shape == (B, *dollar, L_q, d_model)
    assert attn.shape == (B, *dollar, n_head, L_q, L_k)
