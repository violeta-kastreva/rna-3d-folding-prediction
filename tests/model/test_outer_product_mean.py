import pytest
import torch

from model.outer_product_mean import OuterProductMean


@pytest.mark.parametrize("B, L, in_dim, dim_msa, pairwise_dim, dollar",
    [
        (4, 64, 16, 8, 32, dollar)
        for dollar in (tuple(), (3, 2))
    ]
)
def test_outer_product_mean(B, L, in_dim, dim_msa, pairwise_dim, dollar):
    opm = OuterProductMean(
        in_dim=in_dim,
        dim_msa=dim_msa,
        pairwise_dim=pairwise_dim,
    )

    seq_rep = torch.randn(B, *dollar, L, in_dim)
    pair_rep_bias = torch.randn(B, *dollar, L, L, pairwise_dim)
    output = opm(seq_rep, pair_rep_bias)

    assert output.shape == (B, *dollar, L, L, pairwise_dim)