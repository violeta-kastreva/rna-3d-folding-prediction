from typing import Union, Optional, Any

import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F


class SequencePadder:
    def __init__(self, pad_token_id: int):
        self.pad_token_id: int = pad_token_id

    def pad(
            self,
            sequences: Union[list[torch.Tensor], torch.Tensor],
            sequence_axis: int = 0,
            do_calculate_mask: bool = False,
            pad_value: Optional[Any] = None,
            device: Optional[torch.device] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        transpose_condition: bool = sequence_axis != 0
        if isinstance(sequences, torch.Tensor) and transpose_condition:
            # If the input is a single tensor, we need to convert it to a list of tensors
            sequences = sequences.transpose(sequence_axis, 0)

        if isinstance(sequences, torch.Tensor):
            sequences = [sequences[i] for i in range(len(sequences))]

        padded_sequences = pad_sequence(
            sequences,
            batch_first=True,
            padding_value=self.pad_token_id if pad_value is None else pad_value
        ).to(device=device)

        if transpose_condition:
            padded_sequences = padded_sequences.transpose(sequence_axis + 1, 0 + 1)

        if not do_calculate_mask:
            return padded_sequences, None

        mask: torch.Tensor = padded_sequences != self.pad_token_id
        return padded_sequences, mask

    def pad_pack_2_dim_matrices(
            self,
            tensors: list[torch.Tensor],
            pad_value: Optional[Any] = None,
            device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        # tensors: list of (H_i, W_i) tensors
        max_h = max(t.shape[0] for t in tensors)
        max_w = max(t.shape[1] for t in tensors)

        padded = []
        for t in tensors:
            h, w = t.shape
            # pad: (left, right, top, bottom)
            pad = (0, max_w - w, 0, max_h - h)
            padded.append(
                F.pad(t, pad,
                      value=pad_value if pad_value is not None else self.pad_token_id).to(device=device)
            )

        return torch.stack(padded, dim=0).to(device=device)  # Shape: (N, max_h, max_w)
