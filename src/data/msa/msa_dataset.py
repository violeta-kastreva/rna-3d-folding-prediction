from typing import Optional, Callable

import numpy as np
from dataclasses import dataclass

from data.token_encoder import TokenEncoder
from data.msa.msa_loader import MSALoader
from data.msa.msa_similarity_calculator import MSASimilarityCalculator
from data.msa.msa_block_remover import MSABlockRemover
from data.msa.msa_clusterer import MSAClusterer


@dataclass
class MSAConfig:
    block_size_remove_factor: float
    num_blocks_to_remove: int
    min_num_seqs_to_keep: int
    num_representatives: int
    mutation_percent: float


class MSADataset:
    def __init__(
            self,
            msa_folders: list[str],
            msa_config: MSAConfig,
            residues: list[str],
            token_encoder: TokenEncoder,
    ):
        c = msa_config
        self.loader: MSALoader = MSALoader(msa_folders)
        self.similarity_calculator: MSASimilarityCalculator = MSASimilarityCalculator()
        self.block_remover: MSABlockRemover = MSABlockRemover(
            c.block_size_remove_factor,
            c.num_blocks_to_remove,
            c.min_num_seqs_to_keep,
        )
        self.clusterer: MSAClusterer = MSAClusterer(
            c.num_representatives,
            token_encoder.encode(np.array(residues)),
            c.mutation_percent,
        )
        self.token_encoder: TokenEncoder = token_encoder

    def has_msa(self, target_id: str) -> bool:
        """
        Check if the MSA for a given target ID exists.
        """
        return self.loader.has_msa(target_id)

    def get_msa(
            self,
            target_id: str,
            has_msa: bool,
            sequence: str,
            transform: Optional[Callable]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Load the MSA for a given target ID.
        """
        msa: np.ndarray
        if has_msa:
            msa = self.loader.get_msa(target_id, len(sequence))
        else:
            msa = np.array(list(sequence), dtype='U1')

        msa = self.token_encoder.encode(msa)

        if transform is not None:
            msa = transform(msa)

        msa, similarity_percents = self.similarity_calculator.sort(msa)
        msa, ids_to_keep = self.block_remover.remove_blocks(msa)
        similarity_percents = similarity_percents[ids_to_keep]
        cluster_representatives, msa_profiles = self.clusterer.cluster(msa, extra_info=similarity_percents)

        return cluster_representatives, msa_profiles


if __name__ == "__main__":
    root_path_: str = r"E:\Raw Datasets\Stanford RNA Dataset"
    msa_folders_: list[str] = [root_path_ + "\\" + directory for directory in ("MSA", "MSA_v2")]

    target_id_: str = "1A51_A"

    msa_config_: MSAConfig = MSAConfig(
        block_size_remove_factor=0.15,
        num_blocks_to_remove=3,
        min_num_seqs_to_keep=10,
        num_representatives=16,
        mutation_percent=0.15,
    )

    residues_ = list("ACGU-")

    token_encoder_: TokenEncoder = TokenEncoder(np.array(residues_), {x: i for i, x in enumerate(residues_)}, 4)

    msa_dataset_: MSADataset = MSADataset(msa_folders_, msa_config_, residues_, token_encoder_)

    if msa_dataset_.has_msa(target_id_):
        cluster_representatives_, msa_profiles_ = msa_dataset_.get_msa(target_id_, has_msa=True, sequence="A" * 41, transform=None)
        print(f"Cluster Representatives for {target_id_}: {cluster_representatives_.shape}\n{cluster_representatives_}")
        print(f"MSA Profiles for {target_id_}: {msa_profiles_.shape}\n{msa_profiles_}")

    else:
        print(f"No MSA found for target ID: {target_id_}")
