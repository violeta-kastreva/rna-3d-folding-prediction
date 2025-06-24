from typing import Union, Optional, Any

import torch
from torch.nn.utils.rnn import pad_sequence


class SequencePadder:
    def __init__(self, pad_token_id: int):
        self.pad_token_id: int = pad_token_id

    def pad(
            self,
            sequences: Union[list[torch.Tensor], torch.Tensor],
            sequence_axis: int = 0,
            do_calculate_mask: bool = False,
            pad_value: Optional[Any] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert isinstance(sequences, torch.Tensor) and sequence_axis == 0
        transpose_condition: bool = isinstance(sequences, torch.Tensor) and sequence_axis != 0
        if transpose_condition:
            # If the input is a single tensor, we need to convert it to a list of tensors
            sequences = sequences.transpose(sequence_axis, 0)

        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=self.pad_token_id if pad_value is None else pad_value)

        if transpose_condition:
            padded_sequences = padded_sequences.transpose(sequence_axis + 1, 0 + 1)

        if not do_calculate_mask:
            return padded_sequences, None

        mask: torch.Tensor = padded_sequences != self.pad_token_id
        return padded_sequences, mask
