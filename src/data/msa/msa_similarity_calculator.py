import numpy as np


class MSASimilarityCalculator:
    @staticmethod
    def percent_identity(msa: np.ndarray, consensus: np.ndarray) -> np.ndarray:
        """ Compute percent identity to consensus. """
        matches = np.sum(msa == consensus[None, :], axis=-1)
        length = consensus.shape[0]
        return matches / length if length > 0 else np.zeros_like(matches)

    def sort(self, msa: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Sort MSA by percent identity to consensus.
        :param msa: MSA as a 2D numpy array where each row is a sequence.
        Returns:
            tuple: Sorted MSA and corresponding percent identities.
        """
        consensus: np.ndarray = msa[0]
        percent_identity = self.percent_identity(msa, consensus)

        # Sort by identity (descending = best match first)
        sorted_ids = np.argsort(percent_identity, kind="stable")[::-1]
        sorted_msa = msa[sorted_ids]
        percent_identity = percent_identity[sorted_ids]
        assert (sorted_msa[0] == consensus).all(), \
            "Consensus sequence is not the first in the sorted MSA."
        return sorted_msa, percent_identity
