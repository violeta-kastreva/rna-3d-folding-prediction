import torch
from torch import nn

from model.mish_activation import Mish
from model.residual_block import ResidualBlock


class SequenceFeatureExtractor(nn.Module):
    def __init__(
        self,
        embedding: nn.Embedding,
        d_model: int,
        dropout: float,
        d_msa: int,
        d_msa_extra,
    ):
        super().__init__()
        self.embedding: nn.Embedding = embedding
        self.d_model: int = d_model
        self.d_input: int = d_msa * d_model + d_msa_extra

        self.transition = nn.Sequential(
            nn.Linear(self.d_input, self.d_model),
            nn.Dropout(dropout),
            Mish(),
            ResidualBlock(
                nn.LayerNorm(self.d_model),
                nn.Linear(self.d_model, self.d_model),
                nn.Dropout(dropout),
                Mish(),
            ),
        )

    def forward(
            self,
            msa: torch.Tensor,  # Shape: (B, $, L, d_msa)
            msa_profiles: torch.Tensor,  # Shape: (B, $, L, d_msa_extra)
    ) -> torch.Tensor:
        """
        Extracts sequence features from MSA and MSA profiles.
        :param msa:
            A tensor representing multiple sequence alignments (MSA) of shape (B, $, L, d_msa).
            Here, B is the batch size, $ is the number of MSAs, L is the sequence length, and d_msa is the dimension of MSA features.
        :param msa_profiles:
            A tensor representing MSA profiles of shape (B, $, L, d_msa_extra).
        :return:
            A tensor of shape (B, $, L, d_model) representing the extracted sequence features.
        """
        msa = self.embedding(msa)  # Shape: (B, $, L, d_msa, d_model)
        msa = msa.view(*msa.shape[:-2], -1)  # Shape: (B, $, L, d_msa * d_model)

        sequence_data = torch.concatenate([msa, msa_profiles], dim=-1)  # Shape: (B, $, L, d_input)

        sequence_data = self.transition(sequence_data)  # Shape: (B, $, L, d_model)

        return sequence_data

