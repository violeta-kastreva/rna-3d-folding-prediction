import pytest
import torch

from loss.topological_loss import TopologicalLoss


@pytest.mark.parametrize(
    "predicted, data_batch",
    [
        (
                (torch.randn(B, L, d_3d), torch.randn(B, L, d_prob)),
            {
                "sequence_mask": sequence_mask,
                "ground_truth": torch.randn(B, L, num_ground_truths, d_3d),
                "is_synthetic": torch.rand(B) < 0.5,
            },
        )
        for B in (4,)
        for L in (50,)
        for sequence_mask in (torch.rand(B, L) < 0.5,)
        for num_ground_truths in (10,)
        for d_3d in (3,)
        for d_prob in (1,)
    ]
)
def test_forward(predicted, data_batch):
    loss = TopologicalLoss()

    loss, agg_losses, _ = loss(predicted, data_batch)

    assert all(isinstance(l.item(), float) for l in agg_losses + (loss,)), \
        "All losses should be float values."