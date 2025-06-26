import random
from io import StringIO
from typing import Optional

import numpy as np
import pandas as pd
import torch
from overrides import overrides

from data.msa.msa_dataset import MSADataset
from data.msa.msa_loader import MSALoader
from data.rna_dataset_base import RNADatasetBase
from data.sequence_padder import SequencePadder
from data.token_encoder import TokenEncoder
from data.batch_collator import BatchCollator
from data.typing import DataPoint, DataBatch


class ExperimentalDataset(RNADatasetBase):
    def __init__(
            self,
            sequence_files: list[str],
            label_files: list[str],
            msa_dataset: MSADataset,
            sequence_padder: SequencePadder,
            batch_collator: BatchCollator,
            chance_flip: float,
            chance_use_msa_when_available: float,
            device: torch.device,
    ):
        self.df_sequences: pd.DataFrame = pd.concat(
            [pd.read_csv(filepath) for filepath in sequence_files],
            ignore_index=True,
        )
        self.df_sequences = self.df_sequences[~self.df_sequences["target_id"].duplicated(keep="last")]

        self.df_sequences["all_sequences"] = self.df_sequences["all_sequences"].fillna("")

        df_labels: pd.DataFrame = pd.concat(
            [pd.read_csv(filepath) for filepath in label_files],
            ignore_index=True,
        )
        df_labels = df_labels[~df_labels["ID"].duplicated(keep="last")]

        df_labels['target_id'] = df_labels['ID'].str.rsplit('_', n=1).str[0]
        self.df_labels: pd.api.typing.DataFrameGroupBy = df_labels.drop(columns=['ID']).groupby('target_id')

        self.msa_dataset: MSADataset = msa_dataset
        self.encoder: TokenEncoder = msa_dataset.token_encoder
        self.sequence_padder: SequencePadder = sequence_padder
        self.batch_collator: BatchCollator = batch_collator
        self.chance_flip: float = chance_flip
        self.chance_use_msa_when_available: float = chance_use_msa_when_available
        self.device: torch.device = device

    @overrides
    def records_lengths(self) -> list[int]:
        return self.df_sequences["sequence"].apply(len).tolist()

    @overrides
    def __len__(self) -> int:
        return len(self.df_sequences)

    @overrides
    def __getitem__(self, idx: int) -> DataPoint:
        if idx < 0:
            idx += len(self)

        if not (0 <= idx < len(self)):
            raise IndexError("Index out of bounds")

        target_id: str = self.df_sequences["target_id"].iloc[idx]
        sequence, should_reverse = self._get_sequence(idx)
        has_msa, msa, msa_profiles = self._get_msa(target_id, idx, should_reverse)
        num_product_sequences, product_sequences = self._get_product_sequences(idx)
        ground_truth, num_ground_truths, ground_truth_mask = self._get_ground_truth(idx, target_id, should_reverse)

        result: DataPoint = {
            "target_id": target_id,
            "sequence": sequence,
            "sequence_mask": torch.ones_like(sequence, dtype=torch.bool, device=self.device),
            "has_msa": torch.tensor(has_msa, dtype=torch.bool, device=self.device),
            "msa": torch.tensor(msa, dtype=torch.int32, device=self.device),
            "msa_profiles": torch.tensor(msa_profiles, dtype=torch.float32, device=self.device),
            "num_product_sequences": torch.tensor(num_product_sequences, dtype=torch.int32, device=self.device),
            "product_sequences": product_sequences,
            "ground_truth": ground_truth,
            "num_ground_truths": torch.tensor(num_ground_truths, dtype=torch.int32, device=self.device),
            "ground_truth_mask": ground_truth_mask,
            "is_synthetic": torch.tensor(False, dtype=torch.bool, device=self.device),
        }

        return result

    def _get_sequence(self, idx: int) -> tuple[torch.Tensor, bool]:
        sequence: np.ndarray = self.encoder.encode(np.array(list(
            self.df_sequences["sequence"].iloc[idx]), dtype='U1'
        ))
        should_reverse: bool = random.random() < self.chance_flip
        if should_reverse:
            sequence = sequence[::-1].copy()
        return torch.tensor(sequence, dtype=torch.int32, device=self.device), should_reverse

    def _get_msa(self, target_id: str, idx: int, should_reverse: bool) -> tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        has_msa: bool = self.msa_dataset.has_msa(target_id) and random.random() < self.chance_use_msa_when_available

        msa, msa_profiles = self.msa_dataset.get_msa(
            target_id,
            has_msa,
            self.df_sequences["sequence"].iloc[idx],
            lambda x: x[:, ::-1] if should_reverse else x
        )
        return has_msa, msa, msa_profiles

    def _get_product_sequences(self, idx: int) -> tuple[int, Optional[list[torch.Tensor]]]:
        all_sequences: list[list[str]] = self._parse_sequences(self.df_sequences["all_sequences"].iloc[idx])
        num_product_sequences: int = len(all_sequences)
        all_sequences = [
            seq[::-1] if random.random() < self.chance_flip else seq
            for seq in all_sequences
        ]

        if num_product_sequences == 0:
            return num_product_sequences, None

        product_sequences = [
            torch.tensor(
                self.encoder.encode(np.array(list(seq), dtype='U1')),
                dtype=torch.int32,
                device=self.device,
            )
            for seq in all_sequences
        ]

        return num_product_sequences, product_sequences

    @staticmethod
    def _parse_sequences(sequences: str) -> list[list[str]]:
        return MSALoader.parse_fasta(StringIO(sequences))

    def _get_ground_truth(
        self, idx: int, target_id: str, should_reverse: bool
    ) -> tuple[torch.Tensor, int, torch.Tensor]:
        label: pd.DataFrame = self.df_labels.get_group(target_id).copy()

        assert target_id in self.df_labels.groups and not label.empty, \
            f"No GT: {self.df_sequences['target_id'].iloc[idx]}"

        label = (label
                 .drop(columns=["target_id", "resname"])
                 .sort_values(by='resid')
                 .set_index(keys=['resid'])
                 )

        assert len(label.columns) % 3 == 0, (
            "Label columns should be a multiple of 3, but got: {}".format(len(label.columns)))
        num_ground_truths: int = len(label.columns) // 3
        last_index_valid_ground_truth: int = num_ground_truths
        axes: list[str] = ["x", "y", "z"]
        HARDCODED_EPS_FOR_ZERO: float = -1e+18
        for i in range(num_ground_truths):
            if all(
                    (label[f"{coord}_{i + 1}"].isna() | label[f"{coord}_{i + 1}"] == HARDCODED_EPS_FOR_ZERO).all()
                    for coord in axes
            ):
                last_index_valid_ground_truth = i
                break

        num_ground_truths: int = last_index_valid_ground_truth
        ground_truth: torch.Tensor = torch.zeros(
            (len(label), num_ground_truths, 3), dtype=torch.float32, device=self.device,
        )
        for i, coord in enumerate(axes):
            ground_truth[:, :, i] = torch.tensor(
                label[[
                    f"{coord}_{j + 1}"
                    for j in range(num_ground_truths)
                ]].values.T.reshape((len(label), num_ground_truths)),
                dtype=torch.float32,
                device=self.device,
            )

        if should_reverse:
            ground_truth = torch.flip(ground_truth, dims=[0])

        ground_truth_mask = ~torch.isnan(ground_truth) & (ground_truth != HARDCODED_EPS_FOR_ZERO)
        ground_truth[~ground_truth_mask] = 0.0
        ground_truth_mask = ground_truth_mask.all(dim=-1)

        return ground_truth, num_ground_truths, ground_truth_mask

    @overrides
    def collate_fn(self, batch: list[DataPoint]) -> DataBatch:
        return self.batch_collator(batch)


if __name__ == "__main__":
    root_path_: str = r"E:\Raw Datasets\Stanford RNA Dataset"
    # sequence_files_: list[str] = [root_path_ + "\\" + filename for filename in ("train_sequences.csv", "train_sequences.v2.csv",)]
    # label_files_: list[str] = [root_path_ + "\\" + filename for filename in ("train_labels.csv", "train_labels.v2.csv",)]
    sequence_files_: list[str] = [root_path_ + "\\" + filename for filename in ("validation_sequences.csv",)]
    label_files_: list[str] = [root_path_ + "\\" + filename for filename in ("validation_labels.csv",)]
    # sequence_files: list[str] = [root_path + "\\" + filename for filename in ("test_sequences.csv",)]
    # label_files: list[str] = []
    msa_folders_: list[str] = [root_path_ + "\\" + directory for directory in ("MSA", "MSA_v2",)]

    from data.msa.msa_dataset import MSAConfig
    from data.token_library import TokenLibrary
    token_lib_ = TokenLibrary()
    d_msa_ = 16
    msa_dataset_ = MSADataset(
        msa_folders=msa_folders_,
        msa_config=MSAConfig(
            block_size_remove_factor=0.15,
            num_blocks_to_remove=3,
            min_num_seqs_to_keep=10,
            num_representatives=d_msa_,
            mutation_percent=0.15,
        ),  # Assuming MSAConfig is not needed for this example
        residues=list("ACGU-"),
        token_encoder=TokenEncoder(
            tokens=np.array(token_lib_.all_tokens, dtype='U1'),
            map_token_to_id=token_lib_.map_token_to_id,
            missing_token_id=token_lib_.missing_residue_token_id,
        ),
    )
    seq_padder_ = SequencePadder(pad_token_id=token_lib_.pad_token_id)
    batch_collator_ = BatchCollator(sequence_padder=seq_padder_)
    dataset_ = ExperimentalDataset(sequence_files_, label_files_, msa_dataset_, seq_padder_, batch_collator_,
                                   chance_flip=0.5, chance_use_msa_when_available=0.95, device=torch.device("cpu"))
    print(f"Dataset length: {len(dataset_)}")
    print(f"First item: {dataset_[0]}")
    for i_, record_ in enumerate(dataset_):
        if i_ % 100 == 0:
            print(f"Label: {record_.get('target_id')}, Coords: {record_.get('ground_truth', 'No ground truth')}")
            # print(
            #     f"Record {i}: {record['sequence']}, "
            #     f"MSA: {record['msa']}, "
            #     f"All sequences: {record['all_sequences']}, "
            #     f"Coords: {record.get('ground_truth', 'No ground truth')}"
            # )