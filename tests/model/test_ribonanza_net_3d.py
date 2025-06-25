import pytest
import torch
import random

from data.token_library import TokenLibrary
from model.ribonanza_net_3d import RibonanzaNet3D, ModelConfig


@pytest.mark.parametrize(
    "model_config, data_batch",
    [
        (
            ModelConfig(
                d_model=128,
                n_heads=8,
                d_pair_repr=32,
                use_triangular_attention=use_triangular_attention,
                use_bidirectional_lstm= use_bidirectional_lstm,
                d_lstm=16,
                num_lstm_layers=16,
                num_blocks=8,
                use_gradient_checkpoint=use_gradient_checkpoint,
                token_library=TokenLibrary(),
            ),
            {
                "sequence": sequence_mask.int() * torch.randint(0, 5, size=(B, L), dtype=torch.int),
                "sequence_mask": sequence_mask,
                "has_msa": has_msa,
                "msa": has_msa[..., None, None].int() * torch.randint(0, 5, size=(B, L, d_msa), dtype=torch.int),
                "msa_profiles": has_msa[..., None, None].float() * torch.randint(0, 5, size=(B, L, d_msa * 3 * 6)),
                "num_product_sequences": num_product_sequences,
                "product_sequences": [
                    torch.randint(0, 5, size=(random.randint(1, 50),), dtype=torch.int)
                    for _ in range(num_product_sequences.sum().int())
                ],
                "product_sequences_indices": torch.tensor([
                    [i, j]
                    for i, num_seq in enumerate(num_product_sequences.tolist())
                    for j in range(num_seq)
                ]).T,  # Shape: (2, num_product_sequences)
            }
        )
        for use_triangular_attention in (True, False)
        for use_gradient_checkpoint in (True, False)
        for use_bidirectional_lstm in (True, False)
        for B in (4,)
        for L in (50,)
        for sequence_mask in (torch.rand(B, L) < 0.5,)
        for has_msa in (torch.rand(B) < 0.5,)
        for d_msa in (32,)
        for num_product_sequences in (torch.randint(0, 20, size=(B,)),)
    ]
)
def test_forward(model_config, data_batch):
    """
    Test the RibonanzaNet3D model with various configurations and data batches.
    """
    model = RibonanzaNet3D(model_config)
    model.eval()
    with torch.no_grad():
        (output_coords, output_probs), _ = model(data_batch)

    assert output_coords.shape == data_batch["sequence"].shape + (model_config.d_regr_outputs,), \
        "Shape mismatch for output_coords"
    assert output_probs.shape == data_batch["sequence"].shape + (model_config.d_prob_outputs,), \
        "Shape mismatch for output_probs"
