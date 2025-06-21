import random


class MSASampler:
    """
    A sampler for MSA (Multiple Sequence Alignment) data.

    This class is designed to handle the sampling of MSA data from a given dataset.
    It provides methods to retrieve the MSA data for a specific index and to get the total number of MSAs available.

    :param d_msa: The number of MSAs to sample.
    """

    def __init__(self, d_msa: int):
        self.d_msa: int = d_msa

    def sample(self, msa: list[str]) -> list[str]:
        """
        Sample MSA data.

        :param msa: A list of MSA sequences.
        :return: A list of sampled MSA sequences.
        """
        if len(msa) < self.d_msa:
            return msa
        return random.sample(msa, self.d_msa)