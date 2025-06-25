from dataclasses import dataclass
from typing import Optional, cast

import torch
import torch.utils.checkpoint as checkpoint
from torch import nn
from torch.nn.utils.rnn import PackedSequence, pad_sequence, pack_padded_sequence

from data.typing import DataBatch
from data.token_library import TokenLibrary
from model.conv_transformer_encoder import ConvTransformerEncoder
from model.mish_activation import Mish
from model.outer_product_mean import OuterProductMean
from model.product_sequences_lstm import ProductSequencesLSTM
from model.relative_positional_encoding import RelativePositionalEncoding
from model.ribonanza_net import recursive_linear_init
from model.sequence_feature_extractor import SequenceFeatureExtractor
from model.multi_head_attention import MultiHeadAttention


@dataclass
class ModelConfig:
    d_model: int  # == d_input
    n_heads: int
    d_pair_repr: int
    use_triangular_attention: bool
    d_lstm: int  # Dimension of LSTM hidden state for the product sequence representations
    num_lstm_layers: int
    num_blocks: int

    token_library: TokenLibrary

    dropout: float = 0.1

    d_regr_outputs: int = 3  # Number of coordinates to predict (x, y, z)
    d_prob_outputs: int = 1  # Number of probabilities to predict

    rel_pos_enc_clip_value: int = 16

    d_msa: int = 32
    # num_representatives * 3 * (num_residues + (extra_info_size ?? 0 )
    d_msa_extra: int = d_msa * 3 * (5 + 1)

    d_hidden: Optional[int] = None

    use_bidirectional_lstm: bool = True

    use_gradient_checkpoint: bool = False

    def __post_init__(self):
        if self.d_hidden is None:
            self.d_hidden = 4 * self.d_model

        self.d_msa_extra: int = self.d_msa * 3 * (5 + 1)


class RibonanzaNet3D(nn.Module):

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

        self.transformer_encoders: nn.ModuleList = nn.ModuleList()
        self.product_sequences_lstms: nn.ModuleList = nn.ModuleList()
        self.product_sequences_mhas: nn.ModuleList = nn.ModuleList()
        for i in range(cfg.num_blocks):
            if i != cfg.num_blocks - 1:
                k: int = 5
            else:
                k: int = 1

            self.transformer_encoders.append(
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

            self.product_sequences_lstms.append(
                ProductSequencesLSTM(
                    d_input=cfg.d_model if i == 0 else cfg.d_lstm,
                    d_output=cfg.d_lstm,
                    num_layers=cfg.num_lstm_layers,
                    is_bidirectional=cfg.use_bidirectional_lstm,
                    dropout=cfg.dropout,
                )
            )

            self.product_sequences_mhas.append(
                MultiHeadAttention(
                    d_model=cfg.d_model,
                    d_k_model=cfg.d_lstm,
                    n_head=cfg.n_heads,
                    d_k=cfg.d_model // cfg.n_heads,
                    dropout=cfg.dropout,
                )
            )

        scale_factor: float = 1.0
        for i, (t_layer, l_layer, a_layer) in enumerate(
                zip(self.transformer_encoders, self.product_sequences_lstms, self.product_sequences_mhas)
        ):
            scale_factor = 1 / (i + 1) ** 0.5
            recursive_linear_init(t_layer, scale_factor)
            recursive_linear_init(l_layer, scale_factor)
            recursive_linear_init(a_layer, scale_factor)

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

    def forward(
            self,
            data: DataBatch,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[list[torch.Tensor], list[torch.Tensor]]]:
        msa: torch.Tensor = data["msa"]
        msa_profiles: torch.Tensor = data["msa_profiles"]
        sequence_mask: torch.Tensor = data["sequence_mask"]
        packed_product_sequences: PackedSequence = self.encode_product_sequences(data)

        assert not self.config.use_triangular_attention or sequence_mask is not None, \
            "Triangular attention requires a sequence mask."

        sequence_features = self.sequence_features_extractor(msa, msa_profiles)  # Shape: (B, $, L, d_model)
        src = sequence_features

        pairwise_features: torch.Tensor = self.checkpoint_apply(
            self.outer_product_mean, (src,), use_reentrant=True,
        )
        pairwise_features += self.pos_encoder(src)

        attention_weights: list[torch.Tensor] = []
        product_sequences_attention_weights: list[torch.Tensor] = []
        product_sequence_attention_mask = None
        for i, (conv_transformer, lstm, product_sequence_mha) in enumerate(
                zip(self.transformer_encoders, self.product_sequences_lstms, self.product_sequences_mhas)
        ):
            # Apply ConvTransformerEncoder
            src, pairwise_features, attention_weights_i = self.checkpoint_apply(
                conv_transformer,
                (src, pairwise_features, sequence_mask),
                use_reentrant=False,
            )
            attention_weights.append(attention_weights_i)

            # Process product sequences with LSTM
            # per_sequence_representations: Shape: (L+, d_lstm)
            per_sequence_representations, packed_product_sequences, _ = lstm(packed_product_sequences)

            src_diff, product_sequence_mha_attention_weights_i, product_sequence_attention_mask = (
                self.attend_with_product_sequences(
                    cast(MultiHeadAttention, product_sequence_mha),
                    sequence_features=src,  # Shape: (B, $, L, d_model)
                    per_sequence_representations=per_sequence_representations,  # Shape: (B, L', d_lstm)
                    data=data,
                    sequence_mask=sequence_mask,  # Shape: (B, $, L)
                    product_sequence_attention_mask=product_sequence_attention_mask,  # Shape: (B, $, L, L')
                )
            )
            src += src_diff  # Shape: (B, $, L, d_model)
            product_sequences_attention_weights.append(product_sequence_mha_attention_weights_i)

        output_coords = self.coords_head(src)  # Shape: (B, $, L, d_regr_outputs)
        output_probs = self.probs_head(src)  # Shape: (B, $, L, d_prob_outputs)

        return (output_coords, output_probs), (attention_weights, product_sequences_attention_weights)

    def encode_product_sequences(self, data: DataBatch) -> PackedSequence:
        sequences = [self.embedding(seq) for seq in data["product_sequences"]]
        lengths = [len(s) for s in data["product_sequences"]]
        padded_sequences = pad_sequence(sequences, batch_first=True)
        packed_sequences: PackedSequence = pack_padded_sequence(padded_sequences, lengths, batch_first=True, enforce_sorted=False)
        return packed_sequences

    @classmethod
    def activation_checkpoint(cls, layer: nn.Module, inputs: tuple, use_reentrant: bool):
        def custom_forward(*inputs):
            nonlocal layer
            return layer(*inputs)

        return checkpoint.checkpoint(custom_forward, *inputs, use_reentrant=use_reentrant)

    def checkpoint_apply(self, layer: nn.Module, inputs: tuple, use_reentrant: bool):
        if not self.config.use_gradient_checkpoint:
            return layer(*inputs)

        return self.activation_checkpoint(layer, inputs, use_reentrant=use_reentrant)

    def unpack_product_sequences_representations(
            self,
            per_sequence_representations: torch.Tensor,
            data: DataBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Unpacks the per-sequence representations into a tensor of shape (B, L', d_lstm) where L' is the maximum
        number of product sequences per data point in the batch.
        Here, we assume that there is only one batching dimension, due to the nature of the data.
        """
        B = data["sequence"].shape[0]
        max_num_product_sequences: int = data["num_product_sequences"].max().item()
        unpacked_product_sequences = torch.zeros(
            (B, max_num_product_sequences, self.config.d_lstm),
            dtype=per_sequence_representations.dtype,
        )
        unpacked_product_sequences_mask = torch.zeros(
            (B, max_num_product_sequences),
            dtype=torch.bool,
        )
        indices = data["product_sequences_indices"]
        unpacked_product_sequences[indices[0, :], indices[1, :], :] = per_sequence_representations
        unpacked_product_sequences_mask[indices[0, :], indices[1, :]] = True
        return unpacked_product_sequences, unpacked_product_sequences_mask

    def attend_with_product_sequences(
            self,
            product_sequence_mha: MultiHeadAttention,
            sequence_features: torch.Tensor,  # Shape: (B, $, L, d_model)
            per_sequence_representations: torch.Tensor,  # Shape: (B, L', d_lstm)
            data: DataBatch,
            sequence_mask: Optional[torch.Tensor] = None,  # Shape: (B, $, L) or None
            product_sequence_attention_mask: Optional[torch.Tensor] = None,  # Shape: (B, $, L, L') or None
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # unpacked_product_sequences: Shape: (B, L', d_lstm)
        # unpacked_product_sequences_mask: Shape: (B, L')
        unpacked_product_sequences, unpacked_product_sequences_mask = self.unpack_product_sequences_representations(
            per_sequence_representations, data,
        )
        if product_sequence_attention_mask is None:
            product_sequence_attention_mask = (
                sequence_mask[..., None] & unpacked_product_sequences_mask[..., None, :]
                if sequence_mask is not None and unpacked_product_sequences_mask is not None
                else None
            )  # Shape: (B, $, L, L')

        src_diff, product_sequence_mha_attention_weights_i = product_sequence_mha(
            q=sequence_features,  # Shape: (B, $, L, d_model)
            k=unpacked_product_sequences,  # Shape: (B, L', d_lstm)
            v=unpacked_product_sequences,  # Shape: (B, L', d_lstm)
            attn_mask=product_sequence_attention_mask,  # Shape: (B, $, L, L')
            is_attn_mask_1d=False,
        )  # Shape: (B, $, L, d_model)
        return src_diff, product_sequence_mha_attention_weights_i, product_sequence_attention_mask
