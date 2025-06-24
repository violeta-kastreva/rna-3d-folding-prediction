from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.utils.checkpoint as checkpoint
from torch import nn

from data.batch_collator import DataBatch, DataPoint
from data.token_library import TokenLibrary
from model.conv_transformer_encoder import ConvTransformerEncoder
from model.mish_activation import Mish
from model.outer_product_mean import OuterProductMean
from model.relative_positional_encoding import RelativePositionalEncoding
from model.sequence_feature_extractor import SequenceFeatureExtractor


def custom_weight_init(m, scale_factor: float) -> None:
    if isinstance(m, nn.Linear):
        d_model = m.in_features  # Set d_model to the input dimension of the linear layer
        upper = 1.0 / (d_model ** 0.5) * scale_factor
        lower = -upper
        torch.nn.init.uniform_(m.weight, lower, upper)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


def recursive_linear_init(m: nn.Module, scale_factor: float) -> None:
    for child_name, child in m.named_modules():
        if 'gate' not in child_name:
            custom_weight_init(child, scale_factor)


@dataclass
class ModelConfig:
    d_model: int  # == d_input
    n_heads: int
    d_pair_repr: int
    use_triangular_attention: bool
    num_blocks: int

    token_library: TokenLibrary

    dropout: float = 0.1

    d_regr_outputs: int = 3  # Number of coordinates to predict (x, y, z)
    d_prob_outputs: int = 1  # Number of probabilities to predict

    rel_pos_enc_clip_value: int = 16

    d_msa: int = 32
    # num_representatives * 3 * (num_residues + (extra_info_size ?? 0 )
    d_msa_extra = d_msa * 3 * (5 + 1)

    d_hidden: Optional[int] = None

    use_gradient_checkpoint: bool = False

    def __post_init__(self):
        if self.d_hidden is None:
            self.d_hidden = 4 * self.d_model


class RibonanzaNet(nn.Module):

    def __init__(self,config: ModelConfig):
        super().__init__()
        self.config: ModelConfig = config
        cfg = config

        self.embedding = nn.Embedding(
            num_embeddings=len(cfg.token_library.all_tokens),
            embedding_dim=cfg.d_model,
            padding_idx=cfg.token_library.pad_token_id,
        )
        self.sequence_features_extractor: SequenceFeatureExtractor = SequenceFeatureExtractor(
            embedding=self.embedding,
            d_model=cfg.d_model,
            dropout=cfg.dropout,
            d_msa=cfg.d_msa,
            d_msa_extra=cfg.d_msa_extra,
        )

        self.outer_product_mean = OuterProductMean(in_dim=cfg.d_model, dim_msa=cfg.d_msa, pairwise_dim=cfg.d_pair_repr)
        self.pos_encoder = RelativePositionalEncoding(dim=cfg.d_pair_repr, clip_value=cfg.rel_pos_enc_clip_value)

        self.transformer_encoder: nn.ModuleList = nn.ModuleList()
        for i in range(cfg.num_blocks):
            if i != cfg.num_blocks - 1:
                k: int = 5
            else:
                k: int = 1

            self.transformer_encoder.append(
                ConvTransformerEncoder(
                    d_model=cfg.d_model,
                    n_heads=cfg.n_heads,
                    d_pair_repr=cfg.d_pair_repr,
                    use_triangular_attention=cfg.use_triangular_attention,
                    dim_msa=cfg.d_msa,
                    dropout=cfg.dropout,
                    k=k,
                )
            )

        scale_factor: float = 1.0
        for i, layer in enumerate(self.transformer_encoder):
            scale_factor = 1 / (i + 1) ** 0.5
            recursive_linear_init(layer, scale_factor)

        d_coords_hidden: int = cfg.d_model // 2
        self.coords_head = nn.Sequential(
            nn.Linear(cfg.d_model, d_coords_hidden),
            nn.Dropout(cfg.dropout),
            Mish(),
            nn.LayerNorm(d_coords_hidden),
            nn.Linear(d_coords_hidden, d_coords_hidden),
            nn.Dropout(cfg.dropout),
            Mish(),
            nn.Linear(d_coords_hidden, cfg.d_regr_outputs),
        )
        recursive_linear_init(self.coords_head, scale_factor)

        d_probs_hidden: int = cfg.d_model // 4
        self.probs_head = nn.Sequential(
            nn.Linear(cfg.d_model, d_probs_hidden),
            nn.Dropout(cfg.dropout),
            Mish(),
            nn.LayerNorm(d_probs_hidden),
            nn.Linear(d_probs_hidden, d_probs_hidden),
            nn.Dropout(cfg.dropout),
            Mish(),
            nn.Linear(d_probs_hidden, cfg.d_prob_outputs),
        )
        recursive_linear_init(self.coords_head, scale_factor)

    @classmethod
    def activation_checkpoint(cls, layer: nn.Module, inputs: tuple, use_reentrant: bool):
        def custom_forward(*inputs):
            nonlocal layer
            return layer(*inputs)

        return checkpoint.checkpoint(custom_forward, *inputs, use_reentrant=use_reentrant)

    def forward(
            self,
            data: Union[DataBatch, DataPoint],
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], list[torch.Tensor]]:
        msa: torch.Tensor = data["msa"]
        msa_profiles: torch.Tensor = data["msa_profiles"]
        sequence_mask: torch.Tensor = data["sequence_mask"]

        assert not self.config.use_triangular_attention or sequence_mask is not None, \
            "Triangular attention requires a sequence mask."

        sequence_features = self.sequence_features_extractor(msa, msa_profiles)  # Shape: (B, $, L, d_model)
        src = sequence_features

        if self.config.use_gradient_checkpoint:
            pairwise_features = self.activation_checkpoint(self.outer_product_mean, (src,), use_reentrant=True)
        else:
            pairwise_features = self.outer_product_mean(src)

        pairwise_features = pairwise_features + self.pos_encoder(src)

        attention_weights = []
        for i, layer in enumerate(self.transformer_encoder):
            if self.config.use_gradient_checkpoint:
                # Use activation checkpointing for memory efficiency
                src, pairwise_features, attention_weights_i = self.activation_checkpoint(
                    layer,
                    (src, pairwise_features, sequence_mask),
                    use_reentrant=False,
                )
            else:
                # Regular forward pass without checkpointing
                src, pairwise_features, attention_weights_i = layer(src, pairwise_features, sequence_mask)

            attention_weights.append(attention_weights_i)

        output_coords = self.coords_head(src)  # Shape: (B, $, L, d_regr_outputs)
        output_probs = self.probs_head(src)  # Shape: (B, $, L, d_prob_outputs)

        return (output_coords, output_probs), attention_weights
