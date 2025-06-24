import pytest
import torch

from model.conv_transformer_encoder import ConvTransformerEncoder


@pytest.mark.parametrize(
    "B, d_model, n_heads, d_pair_repr, use_triangular_attention, dim_msa, L, dollar, add_attn_mask",
    [
        (2, 16, 4, 8, use_triangular_attention, 10, 12, dollar, add_attn_mask)
        for has_attention_bias_dim_for_heads in (True, False)
        for use_triangular_attention in (True, False)
        for add_attn_mask in (True, False) if not use_triangular_attention or add_attn_mask
        for dollar in (tuple(), (3, 2))
    ]
)
def test_conv_transformer_encoder_forward(
        B, d_model, n_heads, d_pair_repr, use_triangular_attention, dim_msa, L, dollar, add_attn_mask,
):
    cte = ConvTransformerEncoder(
        d_model=d_model,
        n_heads=n_heads,
        d_pair_repr=d_pair_repr,
        use_triangular_attention=use_triangular_attention,
        dim_msa=dim_msa,
        dropout=0.1,
        k=3,
    )

    sequence_features = torch.randn(B, *dollar, L, d_model)
    pairwise_features = torch.randn(B, *dollar, L, L, d_pair_repr)
    if add_attn_mask:
        attn_mask = torch.randn(B, *(1 for _ in dollar), L) < 0.5  # Random attention mask

    output_sequence_features, output_pairwise_features, attn_weights = cte(
        sequence_features=sequence_features,
        pairwise_features=pairwise_features,
        sequence_mask=attn_mask if add_attn_mask else None,
    )

    assert output_sequence_features.shape == (B, *dollar, L, d_model)
    assert output_pairwise_features.shape == (B, *dollar, L, L, d_pair_repr)
    assert attn_weights.shape == (B, *dollar, n_heads, L, L)
