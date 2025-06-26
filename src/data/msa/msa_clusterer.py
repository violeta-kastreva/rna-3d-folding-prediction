from typing import Optional, cast, Union

import numpy as np


class MSAClusterer:
    """
    A class to cluster multiple sequence alignments (MSAs) based on their similarity.
    """

    def __init__(self, num_representatives: int, residues: Union[str, np.ndarray], mutation_percent: float = 0.15):
        """
        Initializes the MSAClusterer with a list of MSAs.

        :param num_representatives: The number of representative sequences to keep in each cluster.
        :param residues: A string of residues to consider in the clustering.
        :param mutation_percent: The percentage of residues to mutate in the representative sequences.
        """
        self.num_representatives: int = num_representatives
        self.residues: Union[str, np.ndarray] = residues
        self.mutation_percent: float = mutation_percent

    def cluster(self, msa: np.ndarray, extra_info: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Clusters the MSAs based on their similarity.

        :return: A list of clusters, where each cluster is a list of MSAs.
        """
        num_clusters: int = min(msa.shape[0], self.num_representatives)
        seq_len: int = msa.shape[1]
        ids_representatives: np.ndarray = np.concatenate((
            [0],
            np.sort(np.random.choice(np.arange(1, msa.shape[0]), size=num_clusters - 1, replace=False)),
        ))
        cluster_representatives = msa[ids_representatives, :]

        cluster_ids_ranges = self.group_by_hamming_distance(msa, ids_representatives)
        clusters_info = np.stack([
            self.aggregate_cluster_info(
                msa[cluster_ids, :],
                msa[id_representative, :],
                extra_info[cluster_ids, ...] if extra_info is not None else None
            )
            for cluster_ids, id_representative
            in zip(cluster_ids_ranges, ids_representatives)
        ], axis=0)  # Shape: (num_clusters, seq_len, 3 * num_residues + 3 * (extra_info_shape ?? 0 ))

        cluster_representatives = self.duplicate_if_needed(
            cluster_representatives, self.num_representatives,
        )  # Shape: (num_representatives, seq_len)
        clusters_info = self.duplicate_if_needed(
            clusters_info, self.num_representatives,
        )  # Shape: (num_representatives, seq_len, 3 * num_residues + 3 * (extra_info_size ?? 0 ))

        clusters_residue_frequencies = clusters_info[:, :, :len(self.residues)]  # Shape: (num_representatives, seq_len, num_residues)
        cluster_representatives = self.mutate_representatives(
            cluster_representatives, clusters_residue_frequencies,
        )  # Shape: (num_representatives, seq_len)

        cluster_representatives = cluster_representatives.T.reshape(seq_len, -1)  # Shape: (seq_len, num_representatives)
        clusters_info = clusters_info.transpose((1, 0, 2)).reshape(seq_len, -1)  # Shape: (seq_len, num_representatives * 3 * (num_residues + (extra_info_size ?? 0 )))
        return cluster_representatives, clusters_info

    @staticmethod
    def group_by_hamming_distance(data: np.ndarray, ids_representatives: np.ndarray) -> list[np.ndarray]:
        groups = [[el] for el in ids_representatives]

        for i, row in enumerate(data):
            distances = [np.sum(row != data[id_rep, :]) for id_rep in ids_representatives]
            closest = np.argmin(distances)
            if i not in ids_representatives:
                groups[closest].append(i)

        groups = [np.array(group, dtype=np.int32) for group in groups]
        return groups

    def aggregate_cluster_info(
            self,
            cluster_msa: np.ndarray,
            representative: np.ndarray,
            extra_cluster_info: np.ndarray,
    ):
        """
        Aggregate information from the cluster MSA.

        Returns NDArray of shape (seq_len, 3 * num_residues) containing:

        - The fraction of each residue in the cluster MSA.
        - The cumulative normalized count of each residue upto that position in all sequences in the cluster MSA.
        - The cumulative normalized count of each residue upto that position in the representative sequence.
        - Additional information from extra_cluster_info if provided, aggregated as max, min, and mean per element.
        """
        num_res: int = len(self.residues)
        ei: Optional[np.ndarray] = (
            extra_cluster_info.reshape((extra_cluster_info.shape[0], -1))
            if extra_cluster_info is not None else None
        )
        aggregate_info = np.zeros(
            (cluster_msa.shape[1], 3 * num_res +
             3 * (ei.shape[1] if ei is not None else 0)),
            dtype=np.float16,
        )
        for i, residue in enumerate(self.residues):
            mask = cluster_msa == residue
            aggregate_info[:, i] = np.sum(mask, axis=0) / cluster_msa.shape[0]
            aggregate_info[:, i + num_res] = self.normalize_range(np.cumsum(np.sum(mask, axis=0)))
            aggregate_info[:, i + 2 * num_res] = self.normalize_range(np.cumsum(representative == residue))

        if ei is not None:
            aggregate_info[:, 3 * num_res::3] = ei.max(axis=0)
            aggregate_info[:, 3 * num_res + 1::3] = ei.min(axis=0)
            aggregate_info[:, 3 * num_res + 2::3] = ei.mean(axis=0)

        return aggregate_info

    @staticmethod
    def normalize_range(vals: np.ndarray) -> np.ndarray:
        """
        Normalize the values to the range [0, 1].
        """
        return 2 / np.pi * np.arctan(vals / 3)

    @staticmethod
    def duplicate_if_needed(arr: np.ndarray, target_length: int) -> np.ndarray:
        """
        Duplicate the array if its length is less than the target length.
        """
        if arr.shape[0] < target_length:
            return np.concatenate((arr,) * (target_length // arr.shape[0] + 1), axis=0)[:target_length, ...]
        return arr[:target_length]

    def mutate_representatives(self, representatives: np.ndarray, residue_frequencies: np.ndarray) -> np.ndarray:
        """
        Mutate the representatives by randomly sampling sequences from the MSA.

        :param representatives: The representative sequences. Shape: (num_clusters, seq_len)
        :param residue_frequencies: The frequencies of residues in the representatives.
            Shape: (num_clusters, seq_len, num_residues)

        :return: The mutated representative sequences.
        """
        position_mask: np.ndarray = cast(np.ndarray, np.random.rand(*representatives.shape) <= self.mutation_percent)
        action_chance: np.ndarray = np.random.rand(*representatives.shape)

        mutated_representatives = np.array([
            [
                np.random.choice(self.residues, p=probabilities / (1.0 if np.isnan(sum_probs) or sum_probs == 0.0 else sum_probs))
                for probabilities in residue_frequencies[i]
                for sum_probs in (np.sum(probabilities),)
            ]
            for i in range(len(residue_frequencies))
        ], dtype=representatives.dtype)

        representatives = np.where(
            position_mask,
            np.where(
                action_chance < 1 / 3,
                np.random.choice(self.residues, size=representatives.shape),
                mutated_representatives,
            ),
            representatives,
        )

        return representatives
