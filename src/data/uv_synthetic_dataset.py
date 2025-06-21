import os
from typing import Any

import torch
from torch.utils.data import Dataset
import pandas as pd

from data.utils import collate_fn


class UVSyntheticDataset(Dataset):
    def __init__(self, root_path: str, index_filepath: str):
        self.root_path: str = root_path
        self.index_df: pd.DataFrame = pd.read_csv(index_filepath, sep=",")

    def records_lengths(self) -> list[int]:
        return self.index_df["seq_len"].tolist()

    def __len__(self):
        return len(self.index_df)

    def __getitem__(self, idx):
        if idx < 0:
            idx += len(self)

        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds for dataset of length {len(self)}")

        row = self.index_df.iloc[idx]
        target_id = row['target_id']

        input = pd.read_csv(os.path.join(self.root_path, f"{target_id}.in"), sep=",").iloc[0]
        label: pd.DataFrame = pd.read_csv(os.path.join(self.root_path, f"{target_id}.gt"), sep=",")

        sequence: str = input['sequence']

        label.drop(columns=['ID', 'target_id'], inplace=True)
        label.sort_values(by='resid', inplace=True)

        ground_truth: torch.Tensor = torch.zeros([len(sequence), 3], dtype=torch.float32)
        for i, coord in enumerate(['x_1', 'y_1', 'z_1']):
            ground_truth[:, i] = torch.tensor(label[coord].values, dtype=torch.float32)

        result: dict[str, Any] = {
            "target_id": target_id,
            "sequence": sequence,
            "msa": [sequence],
            "all_sequences": [sequence],
            "ground_truth": ground_truth,
            "is_synthetic": torch.tensor(True, dtype=torch.bool),
            "msa_atom_index": list(range(len(sequence))),
        }

        return result

    @staticmethod
    def collate_fn(batch):
        return collate_fn(batch)


if __name__ == "__main__":
    root_path = r"E:\Raw Datasets\UW Synthetic RNA structures\converted"
    index_filepath = os.path.join(root_path, "index.csv")

    dataset = UVSyntheticDataset(root_path, index_filepath)

    print(f"Dataset length: {len(dataset)}")
    for i, record in enumerate(dataset):
        if i % 50_000 == 0:
            print("Id: ", record['target_id'],
                  "Ground truth", record['ground_truth'])


