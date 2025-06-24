from typing import Optional

from torch import nn
import torch
from einops import rearrange

from model.convolution import ConvNd
from model.dropout import DropoutRowwise, DropoutColumnwise
from model.mish_activation import Mish
from model.multi_head_attention import MultiHeadAttention
from model.outer_product_mean import OuterProductMean
from model.triangular_multiplicative_module import TriangleMultiplicativeModule
from model.triangular_attention import TriangularAttention


class ConvTransformerEncoder(nn.Module):
    """
    A Transformer Encoder Layer with convolutional enhancements and pairwise feature processing.

    :param d_model: Dimension of the input embeddings
    :param n_heads: Number of attention heads
    :param d_pair_repr: Dimension of pairwise features
    :param use_triangular_attention: Whether to use triangular attention modules
    :param dim_msa: Dimension of the MSA for outer product mean
    :param dropout: Dropout rate
    :param k: Kernel size for the 1D convolution
    """

    def __init__(
            self,
            d_model: int,
            n_heads: int,
            d_pair_repr: int,
            use_triangular_attention: bool,
            dim_msa: int,
            dropout: float = 0.1,
            k: int = 3,
    ):
        super().__init__()

        self.conv = ConvNd(n=1, in_channels=d_model, out_channels=d_model, kernel_size=k, padding=k // 2)
        # === Attention Layers ===
        self.self_attention: MultiHeadAttention = MultiHeadAttention(
            d_model, n_heads, d_k=d_model // n_heads, dropout=dropout
        )

        # === Layer Norms ===
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # === Dropout Layers ===
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.pairwise2heads = nn.Linear(d_pair_repr, n_heads, bias=False)
        self.pairwise_norm = nn.LayerNorm(d_pair_repr)
        self.activation = Mish()

        self.triangle_update_out = TriangleMultiplicativeModule(dim=d_pair_repr, mix='outgoing')
        self.triangle_update_in = TriangleMultiplicativeModule(dim=d_pair_repr, mix='ingoing')

        self.pair_dropout_out = DropoutRowwise(dropout)
        self.pair_dropout_in = DropoutColumnwise(dropout)

        self.use_triangular_attention = use_triangular_attention
        if self.use_triangular_attention:
            self.n_heads_tri_attention: int = 4
            self.triangle_attention_out = TriangularAttention(
                d_model=d_pair_repr,
                n_heads=self.n_heads_tri_attention,
                d_k=d_pair_repr // self.n_heads_tri_attention,
                wise='row'
            )
            self.triangle_attention_in = TriangularAttention(
                d_model=d_pair_repr,
                n_heads=self.n_heads_tri_attention,
                d_k=d_pair_repr // self.n_heads_tri_attention,
                wise='col'
            )

            self.pair_attention_dropout_out = DropoutRowwise(dropout)
            self.pair_attention_dropout_in = DropoutColumnwise(dropout)

        self.outer_product_mean = OuterProductMean(in_dim=d_model, dim_msa=dim_msa, pairwise_dim=d_pair_repr)

        self.pair_transition = nn.Sequential(
            nn.LayerNorm(d_pair_repr),
            nn.Linear(d_pair_repr, d_pair_repr * 4),
            nn.ReLU(),
            nn.Linear(d_pair_repr * 4, d_pair_repr),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Sequence transition is new
        self.sequence_transition = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
            self,
            sequence_features: torch.Tensor,  # Shape: (B, $, L, d_model)
            pairwise_features: torch.Tensor,  # Shape: (B, $, L, L, d_pair_repr)
            sequence_mask: Optional[torch.Tensor] = None,  # Shape: (B, $, L)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the ConvTransformerEncoderLayer.

        :param sequence_features: Sequence feature tensor of shape (B, $, L, d_model)
        :param pairwise_features: Pairwise feature tensor of shape (B, $, L, L, d_pair_repr)
        :param sequence_mask: Optional mask tensor of shape (B, $, L)

        Returns a tuple containing:
            - Updated sequence features of shape (B, $, L, d_model)
            - Updated pairwise features of shape (B, $, L, L, d_pair_repr)
            - Attention weights of shape (B, $, n_heads, L, L) if pairwise_features is provided, otherwise None
        """
        assert not self.use_triangular_attention or sequence_mask is not None, \
            "Triangular attention requires a sequence mask."

        src: torch.Tensor = sequence_features
        if sequence_mask is not None:
            src = src * sequence_mask.float().unsqueeze(-1)  # Shape: (B, $, L, d_model)
        res = self.conv(src.transpose(-1, -2)).transpose(-1, -2)  # Shape: (B, $, L, d_model)
        res = self.activation(res)
        res = self.dropout2(res)
        src = src + res  # Residual connection
        src = self.norm3(src)  # Shape: (B, $, L, d_model)

        # Linear on Pairwise features
        pairwise_bias = rearrange(
            self.pairwise2heads(self.pairwise_norm(pairwise_features)),
            "... i j h -> ... h i j",
        )  # Shape: (B, $, n_heads, L, L)

        # MHA + Pairwise mask
        res, attention_weights = self.self_attention(
            src, src, src,
            attention_bias=pairwise_bias,
            has_attention_bias_dim_for_heads=True,
            attn_mask=sequence_mask,
        )  # Shape: (B, $, L, d_model), (B, $, n_heads, L, L)
        res = self.activation(res)
        res = self.dropout1(res)  # Shape: (B, $, L, d_model)
        src = src + res  # Residual connection
        src = self.norm1(src)  # Shape: (B, $, L, d_model)

        # Position-wise Feedforward according to architecture diagram. Or Sequence transition
        res = self.sequence_transition(src)  # Shape: (B, $, L, d_model)
        src = src + res  # Residual connection
        src = self.norm2(src)  # Shape: (B, $, L, d_model)

        sq_src = pairwise_features
        sq_src = sq_src + self.outer_product_mean(src)  # Shape: (B, $, L, L, d_pair_repr)

        # Triangular update
        sq_src = sq_src + self.pair_dropout_out(
            self.triangle_update_out(
                sq_src, sequence_mask=sequence_mask,
            )
        )  # Shape: (B, $, L, L, d_pair_repr)
        sq_src = sq_src + self.pair_dropout_in(
            self.triangle_update_in(
                sq_src, sequence_mask=sequence_mask,
            )
        )  # Shape: (B, $, L, L, d_pair_repr)

        # Triangular attention
        if self.use_triangular_attention:
            sq_src = sq_src + self.pair_attention_dropout_out(
                self.triangle_attention_out(
                    sq_src, sequence_mask=sequence_mask,
                )
            )  # Shape: (B, $, L, L, d_pair_repr)
            sq_src = sq_src + self.pair_attention_dropout_in(
                self.triangle_attention_in(
                    sq_src, sequence_mask=sequence_mask,
                )
            )  # Shape: (B, $, L, L, d_pair_repr)

        sq_src = sq_src + self.pair_transition(sq_src)    # Shape: (B, $, L, L, d_pair_repr)

        sequence_features = src  # Shape: (B, $, L, d_model)
        pairwise_features = sq_src  # Shape: (B, $, L, L, d_pair_repr)
        # attention_weights: Shape: (B, $, n_heads, L, L)
        return sequence_features, pairwise_features, attention_weights
