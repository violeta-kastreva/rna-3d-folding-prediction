import pytest
import torch

from model.triangular_attention import TriangularAttention


@pytest.mark.parametrize("B, L, d_model, n_heads, d_k, dollar, wise", [
    (4, 64, 16, 8, 4, dollar, wise)
    for dollar in (tuple(), (3, 2))
    for wise in ("row", "col")
])
def test_forward(B, L, d_model, n_heads, d_k, dollar, wise):
    tr_attn = TriangularAttention(
        d_model=d_model,
        d_k=d_k,
        n_heads=n_heads,
        wise=wise,
    )

    z = torch.randn(B, *dollar, L, L, d_model)  # Shape: (B, $, I, J, d_model)
    sequence_mask = torch.ones(B, *(1 for _ in dollar), L, dtype=torch.bool)  # Shape: (B, $$, I)
    output = tr_attn(z, sequence_mask)  # Shape: (B, $, I, J, d_model)
    assert output.shape == (B, *dollar, L, L, d_model)
