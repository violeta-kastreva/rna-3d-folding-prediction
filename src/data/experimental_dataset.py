import os
from glob import glob
from io import StringIO
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset
from Bio import SeqIO

from data.utils import collate_fn


class ExperimentalDataset(Dataset):
    def __init__(
            self,
            sequence_files: list[str],
            label_files: list[str],
            msa_folders: list[str] = None,
            has_ground_truth: bool = True
    ):
        self.df_sequences: pd.DataFrame = pd.concat(
            [pd.read_csv(filepath) for filepath in sequence_files],
            ignore_index=True,
        )
        self.df_sequences = self.df_sequences[~self.df_sequences["target_id"].duplicated(keep="last")]

        self.df_sequences['all_sequences'] = self.df_sequences['all_sequences'].fillna("")

        self.has_ground_truth: bool = has_ground_truth
        if self.has_ground_truth:
            df_labels: pd.DataFrame = pd.concat(
                [pd.read_csv(filepath) for filepath in label_files],
                ignore_index=True,
            )
            df_labels = df_labels[~df_labels["ID"].duplicated(keep="last")]

            df_labels['target_id'] = df_labels['ID'].str.rsplit('_', n=1).str[0]
            self.df_labels: pd.api.typing.DataFrameGroupBy = df_labels.drop(columns=['ID']).groupby('target_id')
        else:
            self.df_labels: pd.api.typing.DataFrameGroupBy = pd.DataFrame(
                columns=["target_id","resname","resid","x_1","y_1","z_1"]
            ).groupby('target_id')

        self.msa_folders: list[str] = msa_folders
        self.msa_indices: list[str] = [
            filepath.replace("\\", "/").split('/')[-1].removesuffix(".MSA.fasta")
            for msa_folder in msa_folders
            for filepath in glob(f"{msa_folder}/*/*.MSA.fasta")
        ]

    def records_lengths(self) -> list[int]:
        return self.df_sequences["sequence"].apply(len).tolist()

    def __len__(self):
        return len(self.df_sequences)

    def __getitem__(self, idx):
        if idx < 0:
            idx += len(self)

        if not (0 <= idx < len(self)):
            raise IndexError("Index out of bounds")

        sequence: str = self.df_sequences["sequence"].iloc[idx]
        msa: list[str] = self.get_msa(idx) if self._has_msa(idx) else [self.df_sequences["sequence"].iloc[idx]]
        all_sequences: list[str] = self._parse_fasta(
            self.df_sequences["all_sequences"].iloc[idx],
            is_filepath=False
        )
        if len(all_sequences) == 0:
            all_sequences = [sequence]

        assert not self.has_ground_truth or self.df_sequences["target_id"].iloc[idx] in self.df_labels.groups
        assert min(msa, key=len, default=0) == max(msa, key=len, default=0)
        assert len(sequence) <= len(msa[0])

        msa_atom_index: list[int] = []
        ind: int = 0
        for i, letter in enumerate(msa[0]):
            if ind < len(sequence) and letter == sequence[ind]:
                msa_atom_index.append(i)
                ind += 1

        ground_truth: torch.Tensor = torch.zeros((0, 0, 3), dtype=torch.float32)
        if not self.has_ground_truth:
            ground_truth = torch.zeros((0, 0, 3), dtype=torch.float32)
        else:
            label: pd.DataFrame = self.df_labels.get_group(self.df_sequences["target_id"].iloc[idx]).copy()

            assert self.df_sequences["target_id"].iloc[idx] in self.df_labels.groups and not label.empty, \
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
            for i in range(num_ground_truths):
                if any((label[f"{coord}_{i + 1}"].isna() | label[f"{coord}_{i + 1}"] == -1e+18).any() for coord in ['x', 'y', 'z']):
                    last_index_valid_ground_truth = i
                    break

            if self.df_sequences["target_id"].iloc[idx] == "2BQ5_S":
                print("Ha")
            coords: torch.Tensor = torch.zeros((last_index_valid_ground_truth, len(label), 3))
            for i, coord in enumerate(['x', 'y', 'z']):
                coords[:, :, i] = torch.tensor(
                    label[[f"{coord}_{j + 1}" for j in range(last_index_valid_ground_truth)]].values.T,
                    dtype=torch.float32,
                )

            ground_truth = coords

        result: dict[str, Any] = {
            "target_id": self.df_sequences["target_id"].iloc[idx],
            "sequence": sequence,
            "msa": msa,
            "all_sequences": all_sequences,
            "ground_truth": ground_truth,
            "is_synthetic": torch.tensor(False, dtype=torch.bool),
            "msa_atom_index": msa_atom_index,
        }

        return result

    def _has_msa(self, idx):
        # Placeholder for actual MSA check logic
        return self.df_sequences["target_id"].iloc[idx] in self.msa_indices

    @staticmethod
    def _parse_fasta(fasta: str, is_filepath: bool) -> list[str]:
        if not is_filepath:
            if not isinstance(fasta, str):
                print("Fasta: ", fasta)
            fasta = StringIO(fasta)
        return [str(record.seq) for record in SeqIO.parse(fasta, "fasta")]

    def get_msa(self, idx):
        for folder in self.msa_folders:
            msa_file = rf"{folder}\{self.df_sequences['target_id'].iloc[idx]}.MSA.fasta"
            if os.path.exists(msa_file):
                return self._parse_fasta(msa_file, is_filepath=True)

        raise ValueError(f"No MSA for {self.df_sequences['target_id'].iloc[idx]}.")

    @staticmethod
    def collate_fn(batch):
        return collate_fn(batch)


if __name__ == "__main__":
    root_path: str = r"E:\Raw Datasets\Stanford RNA Dataset"
    # sequence_files: list[str] = [root_path + "\\" + filename for filename in ("train_sequences.csv", "train_sequences.v2.csv")]
    # label_files: list[str] = [root_path + "\\" + filename for filename in ("train_labels.csv", "train_labels.v2.csv")]
    sequence_files: list[str] = [root_path + "\\" + filename for filename in ("validation_sequences.csv",)]
    label_files: list[str] = [root_path + "\\" + filename for filename in ("validation_labels.csv")]
    # sequence_files: list[str] = [root_path + "\\" + filename for filename in ("test_sequences.csv",)]
    # label_files: list[str] = []
    msa_folders: list[str] = [root_path + "\\" + directory for directory in ("MSA", "MSA_v2")]

    dataset = ExperimentalDataset(sequence_files, label_files, msa_folders, has_ground_truth=False)
    print(f"Dataset length: {len(dataset)}")
    print(f"First item: {dataset[0]}")
    for i, record in enumerate(dataset):
        if i % 100 == 0:
            print(f"Label: {record.get('target_id')}, Coords: {record.get('ground_truth', 'No ground truth')}")
            # print(
            #     f"Record {i}: {record['sequence']}, "
            #     f"MSA: {record['msa']}, "
            #     f"All sequences: {record['all_sequences']}, "
            #     f"Coords: {record.get('ground_truth', 'No ground truth')}"
            # )