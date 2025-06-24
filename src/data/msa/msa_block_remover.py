from random import choices
import numpy as np


class MSABlockRemover:
    def __init__(self, block_size_factor: float = 0.15, num_blocks: int = 3, min_to_keep: int = 10):
        self.block_size_factor: float = block_size_factor
        self.num_blocks: int = num_blocks
        self.min_to_keep: int = min_to_keep

    def remove_blocks(self, msa: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Remove random blocks of sequences from the MSA.
        This method randomly selects a number of blocks of sequences to remove from the MSA.
        The size of each block is determined by the `block_size_factor` relative to the total number of sequences in
        the MSA, ensuring that the remaining number of sequences is at least `min_to_keep` sequences.
        The number of blocks to remove is specified by `num_blocks`.
        The method returns the modified MSA and the indices of the sequences that were kept.
        """
        N_all_seq: int = msa.shape[0]
        block_size: int = self.calc_block_size(msa)
        ids_to_delete = choices(range(N_all_seq - block_size), k=self.num_blocks)
        ids_to_delete = [[pos, pos + block_size] for pos in ids_to_delete]
        ids_to_delete = self.union_intervals(ids_to_delete)
        ids_to_keep = np.array([
            num
            for [_, a], [b, _] in zip([[-1, 0]] + ids_to_delete, ids_to_delete + [[N_all_seq, -1]])
            for num in range(a, b)
        ], dtype=np.int16)
        msa = msa[ids_to_keep, :]
        return msa, ids_to_keep

    def calc_block_size(self, msa: np.ndarray) -> int:
        """
        Ensure that the block size is at least `min_to_keep` sequences.
        If the calculated block size is less than `min_to_keep`, it will be set to `min_to_keep`.
        """
        N_all_seq: int = msa.shape[0]
        block_size: int = min(
            int(self.block_size_factor * N_all_seq),
            int((N_all_seq - self.min_to_keep) / self.num_blocks)
        )
        return block_size

    @staticmethod
    def union_intervals(intervals: list[list[int]]) -> list[list[int]]:
        """
        Given a list of intervals [start, end), return the union as a list of non-overlapping intervals.
        """
        if not intervals:
            return []

        # Sort intervals by start time
        intervals.sort(key=lambda x: x[0])

        merged = [intervals[0]]

        for current in intervals[1:]:
            last = merged[-1] # Get the last merged interval, as reference
            if current[0] <= last[1]:  # Overlapping or adjacent
                last[1] = max(last[1], current[1])  # Merge
            else:
                merged.append(current)

        return merged

