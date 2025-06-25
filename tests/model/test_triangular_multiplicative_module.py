import pytest
import torch

from model.triangular_multiplicative_module import TriangleMultiplicativeModule



@pytest.mark.parametrize("d_model, L, B, dollar, mix", [
    (16, 64, 4, dollar, mix)
    for dollar in (tuple(), (3, 2))
    for mix in ("ingoing", "outgoing")
])
def test_forward(d_model: int, L: int, B: int, dollar: tuple[int, ...], mix: str):
    tr_mm = TriangleMultiplicativeModule(
        dim=d_model,
        hidden_dim=d_model // 2,
        mix=mix,
    )

    z = torch.randn(B, *dollar, L, L, d_model)
    sequence_mask = torch.ones(B, *(1 for _ in dollar), L, dtype=torch.bool)
    output = tr_mm(z, sequence_mask)
    assert output.shape == (B, *dollar, L, L, d_model)
