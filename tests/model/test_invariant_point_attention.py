import pytest
import torch

from model.invariant_point_attention import InvariantPointAttention

@pytest.mark.parametrize("B, L, d_model, n_heads, d_sk, d_pk, d_sv, d_pv, d_pair_repr, does_require_pair_repr, dollar", [
    (2, 5, 16, 4, 8, 6, 8, 6, 10, True, dollar)
    for dollar in (tuple(), (3, 2))
])
def test_forward(
        B, L, d_model, n_heads, d_sk, d_pk, d_sv, d_pv, d_pair_repr, does_require_pair_repr, dollar
):
    ipa = InvariantPointAttention(
        d_model=d_model,
        n_heads=n_heads,
        d_sk=d_sk,
        d_pk=d_pk,
        d_sv=d_sv,
        d_pv=d_pv,
        d_pair_repr=d_pair_repr,
        does_require_pair_repr=does_require_pair_repr,
    )

    single_repr = torch.randn(B, *dollar, L, d_model)
    pairwise_repr = torch.randn(B, *dollar, L, L, d_pair_repr)
    rotations = torch.randn(B, *dollar, L, InvariantPointAttention.DIMS, InvariantPointAttention.DIMS)
    translations = torch.randn(B, *dollar, L, InvariantPointAttention.DIMS)

    output = ipa(
        single_repr=single_repr,
        pairwise_repr=pairwise_repr,
        rotations=rotations,
        translations=translations,
    )

    assert output.shape == (B, *dollar, L, d_model)
