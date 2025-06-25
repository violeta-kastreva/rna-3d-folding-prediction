import pytest
import torch

from data.token_library import TokenLibrary
from model.ribonanza_net import RibonanzaNet, ModelConfig


@pytest.mark.parametrize(
    "d_model, n_heads, d_pair_repr, d_msa, num_blocks, use_triangular_attention, use_gradient_checkpoint, "
    "data_input",
    [
        (
            d_model, 16, d_pair_repr, d_msa, 9, use_triangular_attention, use_gradient_checkpoint,
            {
                "msa": torch.randint(0, 10, size=(B, *dollar, L, d_msa)),
                "msa_profiles": torch.randn(B, *dollar, L, d_msa * 3 * (5 + 1)),
                "sequence_mask": torch.randn(B, *dollar, L) < 0.5  # Random attention mask
            },
        )
        for B in (2,)
        for L in (128,)
        for d_model in (128,)
        for d_pair_repr in (32,)
        for d_msa in (32,)
        for use_triangular_attention in (True, False)
        for use_gradient_checkpoint in (True, False)
        for dollar in (tuple(), (1, 2))
    ]
)
def test_forward(
        d_model, n_heads, d_pair_repr, d_msa, num_blocks, use_triangular_attention, use_gradient_checkpoint,
        data_input,
):
    cfg = ModelConfig(
        d_model=d_model,
        n_heads=n_heads,
        d_pair_repr=d_pair_repr,
        num_blocks=num_blocks,
        use_triangular_attention=use_triangular_attention,
        token_library=TokenLibrary(),
        use_gradient_checkpoint=use_gradient_checkpoint,
    )
    model: RibonanzaNet = RibonanzaNet(config=cfg)
    (output_coords, output_probs), attention_weights = model(data_input)

    assert output_coords.shape == data_input["msa"].shape[:-1] + (cfg.d_regr_outputs,)
    assert output_probs.shape == data_input["msa"].shape[:-1] + (cfg.d_prob_outputs,)
