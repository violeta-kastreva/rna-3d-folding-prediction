import pytest

from data.msa.msa_block_remover import MSABlockRemover


@pytest.mark.parametrize(
    "input, output",
    [
        ([[1, 3], [2, 4], [3, 6]], [[1, 6]]),
        ([[1, 2], [3, 5], [6, 8]], [[1, 2], [3, 5], [6, 8]]),
        ([[1, 4], [2, 5], [3, 7]], [[1, 7]]),
        ([[1, 2], [1, 4], [5, 6]], [[1, 4], [5, 6]]),
        ([[1, 5], [3, 7], [8, 10], [2, 6], [11, 13], [14, 18], [9, 11]], [[1, 7], [8, 13], [14, 18]]),
    ]
)
def test_union_of_intervals(input, output):
    calc_output = MSABlockRemover.union_intervals(input)
    assert calc_output == output
