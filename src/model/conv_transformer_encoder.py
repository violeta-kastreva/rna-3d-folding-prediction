from typing import Union

from torch import nn
import torch

from model.dropout import DropoutColumnwise
from model.multi_head_attention import MultiHeadAttention
from model.outer_product_mean import OuterProductMean
from model.triangular_multiplicative_module import TriangleMultiplicativeModule
from model.triangular_attention import TriangularAttention


class ConvTransformerEncoderLayer(nn.Module):
    """
    A Transformer Encoder Layer with convolutional enhancements and pairwise feature processing.
    """

    def __init__(
            self,
            d_model: int,
            n_heads: int,
            d_feedforward: int,
            pairwise_dimension: int,
            use_triangular_attention: bool,
            dim_msa: int,
            dropout: float = 0.1,
            k: int = 3,
    ):
        """
        :param d_model: Dimension of the input embeddings
        :param n_heads: Number of attention heads
        :param d_feedforward: Hidden layer size in feedforward network
        :param pairwise_dimension: Dimension of pairwise features
        :param use_triangular_attention: Whether to use triangular attention modules
        :param dropout: Dropout rate
        :param k: Kernel size for the 1D convolution
        """
        super().__init__()

        # === Attention Layers ===
        self.self_attention = MultiHeadAttention(d_model, n_heads, d_k=d_model // n_heads, dropout=dropout)

        # self.linear1 = nn.Linear(d_model, d_feedforward)
        # self.dropout = nn.Dropout(dropout)
        # self.linear2 = nn.Linear(d_feedforward, d_model)

        # === Layer Norms ===
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # self.norm3 = nn.LayerNorm(d_model)

        # === Dropout Layers ===
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        # self.dropout3 = nn.Dropout(dropout)

        self.pairwise2heads = nn.Linear(pairwise_dimension, n_heads, bias=False)
        self.pairwise_norm = nn.LayerNorm(pairwise_dimension)
        self.activation = nn.GELU()

        # self.conv = nn.Conv1d(d_model, d_model, k, padding=k // 2)

        self.triangle_update_out = TriangleMultiplicativeModule(dim=pairwise_dimension, mix='outgoing')
        self.triangle_update_in = TriangleMultiplicativeModule(dim=pairwise_dimension, mix='ingoing')

        self.pair_dropout_out = DropoutRowwise(dropout)
        self.pair_dropout_in = DropoutRowwise(dropout)

        self.use_triangular_attention = use_triangular_attention
        if self.use_triangular_attention:
            self.n_heads_tri_attention: int = 4
            self.triangle_attention_out = TriangularAttention(
                d_model=pairwise_dimension,
                n_heads=self.n_heads_tri_attention,
                d_k=pairwise_dimension // self.n_heads_tri_attention,
                wise='row'
            )
            self.triangle_attention_in = TriangularAttention(
                d_model=pairwise_dimension,
                n_heads=self.n_heads_tri_attention,
                d_k=pairwise_dimension // self.n_heads_tri_attention,
                wise='col'
            )

            self.pair_attention_dropout_out = DropoutRowwise(dropout)
            self.pair_attention_dropout_in = DropoutColumnwise(dropout)

        self.outer_product_mean = OuterProductMean(in_dim=d_model, dim_msa=dim_msa, pairwise_dim=pairwise_dimension)

        self.pair_transition = nn.Sequential(
            nn.LayerNorm(pairwise_dimension),
            nn.Linear(pairwise_dimension, pairwise_dimension * 4),
            nn.ReLU(inplace=True),
            nn.Linear(pairwise_dimension * 4, pairwise_dimension)
        )
        # Sequence transition is new
        self.sequence_transititon = nn.Sequential(nn.Linear(d_model, d_model * 4),
                                                  nn.ReLU(),
                                                  nn.Linear(d_model * 4, d_model))

    def forward(
            self,
            input
    ) -> Union[tuple[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the ConvTransformerEncoderLayer.

        :param src: Input tensor of shape (B, $, L, d_model)
        :param pairwise_repr: Pairwise feature tensor of shape (B, $, L, L, d_pair_repr)
        :param src_mask: Optional mask tensor of shape (batch_size, seq_len)
        :param return_aw: Whether to return attention weights
        :return: Tuple containing processed src and pairwise_features (and optionally attention weights)
        """
        sequence_features, pairwise_features, src_mask, return_aw = input

        use_gradient_checkpoint = False

        # src = src * src_mask.float().unsqueeze(-1)  # Shape: (batch_size, seq_len, d_model)
        # res = src  # residual
        # # 1D convolution
        # src = src + self.conv(src.permute(0, 2, 1)).permute(0, 2, 1)  # Shape: (batch_size, seq_len, d_model)
        # src = self.norm3(src)

        # Linear on Pairwise features
        pairwise_bias = self.pairwise2heads(self.pairwise_norm(pairwise_features)).permute(0, 3, 1,
                                                                                           2)  # Shape: (batch_size, n_head, seq_len, seq_len)
        # MHA + Pairwise mask
        src2, attention_weights = self.self_attention(src, src, src, mask=pairwise_bias,
                                                 src_mask=src_mask)  # Shape: (batch_size, seq_len, d_model)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Position-wise Feedforward according to architecture diagram. Or Sequence transition
        res = src
        src = self.sequence_transititon(src)
        src = res + self.dropout2(src)
        src = self.norm2(src)

        pairwise_features = pairwise_features + self.outer_product_mean(
            src)  # Shape: (batch_size, seq_len, seq_len, pairwise_dimension)
        # Triangular update
        pairwise_features = pairwise_features + self.pair_dropout_out(
            self.triangle_update_out(pairwise_features, src_mask))
        pairwise_features = pairwise_features + self.pair_dropout_in(
            self.triangle_update_in(pairwise_features, src_mask))

        if self.use_triangular_attention:
            pairwise_features = pairwise_features + self.pair_attention_dropout_out(
                self.triangle_attention_out(pairwise_features, src_mask))
            pairwise_features = pairwise_features + self.pair_attention_dropout_in(
                self.triangle_attention_in(pairwise_features, src_mask))

        pairwise_features = pairwise_features + self.pair_transition(
            pairwise_features)  # Shape: (batch_size, seq_len, seq_len, pairwise_dimension)

        if return_aw:
            return src, pairwise_features, attention_weights  # Shapes: (batch_size, seq_len, d_model), (batch_size, seq_len, seq_len, pairwise_dimension), (batch_size, n_heads, seq_len, seq_len)
        else:
            return src, pairwise_features  # Shapes: (batch_size, seq_len, d_model), (batch_size, seq_len, seq_len, pairwise_dimension)