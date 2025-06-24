import os
from glob import glob
from io import StringIO
from typing import Union

import numpy as np
from Bio import SeqIO


class MSALoader:
    """
    A class to load MSA (Multiple Sequence Alignment) data from specified folders.

    This class is designed
    to handle the loading of MSA data from a list of folders, parsing the MSA FASTA files,
    and providing methods to retrieve MSA sequences for specific indices.
    """
    def __init__(self, msa_folders: list[str]):
        self.msa_folders: list[str] = msa_folders
        self.msa_indices: list[str] = [
            filepath.replace("\\", "/").split('/')[-1].removesuffix(".MSA.fasta")
            for msa_folder in msa_folders
            for filepath in glob(f"{msa_folder}/*.MSA.fasta")
        ]

    def has_msa(self, target_id: str) -> bool:
        return target_id in self.msa_indices

    @staticmethod
    def parse_fasta(fasta: Union[str, StringIO]) -> list[list[str]]:
        """
        Parse a FASTA file and return a list of sequences, where each sequence is a list of characters.
        """
        return [list(record.seq) for record in SeqIO.parse(fasta, "fasta")]

    def _clean_msa(self, msa: np.ndarray) -> np.ndarray:
        """
        Remove columns with gaps in the first sequence
        """
        return msa[:, msa[0] != '-']

    def get_msa(self, target_id: str, seq_len: int) -> np.ndarray:
        for folder in self.msa_folders:
            msa_file = rf"{folder}\{target_id}.MSA.fasta"
            if os.path.exists(msa_file):
                clean_msa = self._clean_msa(
                    np.array(self.parse_fasta(msa_file), dtype='U1')
                )
                assert clean_msa.shape[1] == seq_len, \
                    f"MSA length mismatch for {target_id}: expected {seq_len}, got {clean_msa.shape[0]}"
                return clean_msa

        raise ValueError(f"No MSA for {target_id}.")
