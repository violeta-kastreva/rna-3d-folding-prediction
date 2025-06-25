import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence


class ProductSequencesLSTM(nn.Module):
    def __init__(self, d_input: int, d_output: int, num_layers: int, is_bidirectional: bool, dropout: float):
        super().__init__()

        assert d_output % 2 == 0, "Output dimension must be divisible by 2."
        if is_bidirectional:
            d_output = d_output // 2

        self.lstm = nn.LSTM(
            input_size=d_input,
            hidden_size=d_output,
            num_layers=num_layers,
            bidirectional=is_bidirectional,
            batch_first=True,
            dropout=dropout,
        )

    def forward(self, packed_input: PackedSequence) -> tuple[torch.Tensor, PackedSequence, tuple[torch.Tensor, torch.Tensor]]:
        packed_output, (h_n, c_n) =self.lstm(packed_input)

        # Unpack the packed sequence
        per_sequence_representations = self.extract_sequence_representations(packed_output)
        return per_sequence_representations, packed_output, (h_n, c_n)

    def extract_sequence_representations(self, packed_output: PackedSequence) -> torch.Tensor:
        """
        Extracts sequence representations from the packed output of the LSTM.
        """
        # Unpack the packed sequence
        # output: [batch_size, max_seq_len, hidden_size * num_directions]
        # lengths: [batch_size]  # actual lengths of each sequence
        output, lengths = pad_packed_sequence(packed_output, batch_first=True)

        # Get the index of the last valid element for each sequence
        last_indices = lengths - 1  # Shape: [batch_size]

        # Select batch indices
        batch_indices = torch.arange(output.size(0))  # Shape: [batch_size]

        if self.lstm.bidirectional:
            hidden_size: int = self.lstm.hidden_size

            # Forward and backward outputs
            forward_output = output[:, :, :hidden_size]  # Shape: [batch_size, seq_len, hidden_size]
            backward_output = output[:, :, hidden_size:]  # Shape: [batch_size, seq_len, hidden_size]

            # Final forward LSTM outputs
            final_forward = forward_output[batch_indices, last_indices]  # Shape: [batch_size, hidden_size]

            # Final backward LSTM outputs (always first timestep)
            final_backward = backward_output[:, 0, :]  # Shape: [batch_size, hidden_size]

            final_outputs = torch.concatenate(
                [final_forward, final_backward], dim=1,
            )  # Shape: [batch_size, 2 * hidden_size]
        else:
            # Final hidden states per sequence
            final_outputs = output[batch_indices, last_indices]  # Shape: [batch_size, hidden_size]

        return final_outputs
