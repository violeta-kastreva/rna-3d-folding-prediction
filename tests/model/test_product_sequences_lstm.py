import pytest
import torch
import random

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from model.product_sequences_lstm import ProductSequencesLSTM


@pytest.mark.parametrize(
    "batch_size, max_L, all_sequences, lengths, d_input, d_output, num_layers, is_bidirectional",
    [
        (batch_size, max(lengths), all_sequences, lengths, d_input, d_output, num_layers, is_bidirectional)
        for d_input in (14, 28)
        for d_output in (36, 72)
        for num_layers in (1, 3, 5)
        for is_bidirectional in (True, False)
        for batch_size in (16, 32)
        for max_L in (15, 50)
        for lengths in ([random.randint(1, max_L) for _ in range(batch_size)],)
        for all_sequences in ([torch.rand(l, d_input) for l in lengths],)
    ]
)
def test_product_sequences_lstm(
        batch_size, max_L, all_sequences, lengths,
        d_input, d_output, num_layers, is_bidirectional,
):
    num_dirs: int = 2 if is_bidirectional else 1
    padded_sequences = pad_sequence(all_sequences, batch_first=True)
    packed_input = pack_padded_sequence(padded_sequences, lengths, batch_first=True, enforce_sorted=False)

    lstm = ProductSequencesLSTM(
        d_input=d_input, d_output=d_output, num_layers=num_layers, is_bidirectional=is_bidirectional
    )
    per_sequence_representations, packed_output, (h_n, c_n) = lstm(packed_input)

    unpacked_output, _ = pad_packed_sequence(packed_output, batch_first=True)

    assert per_sequence_representations.shape == (batch_size, d_output), \
        "Shape mismatch for per_sequence_representations"
    assert unpacked_output.shape == (batch_size, max_L, d_output), "Shape mismatch for unpacked_output"
    assert h_n.shape == (num_layers * num_dirs, batch_size, d_output // num_dirs), "Shape mismatch for h_n"
    assert c_n.shape == (num_layers * num_dirs, batch_size, d_output // num_dirs), "Shape mismatch for c_n"
