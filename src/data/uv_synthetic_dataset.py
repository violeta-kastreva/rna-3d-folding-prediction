import os

import numpy as np
import torch
from overrides import overrides
import pandas as pd

from data.batch_collator import BatchCollator
from data.msa.msa_dataset import MSADataset
from data.typing import DataPoint, DataBatch
from data.rna_dataset_base import RNADatasetBase
from data.token_encoder import TokenEncoder


class UVSyntheticDataset(RNADatasetBase):
    def __init__(
            self,
            root_path: str,
            index_filepath: str,
            msa_dataset: MSADataset,
            batch_collator: BatchCollator,
            encoder: TokenEncoder,
            chance_flip: float,
            device: torch.device,
    ):
        self.root_path: str = root_path
        self.index_df: pd.DataFrame = pd.read_csv(index_filepath, sep=",")
        self.msa_dataset: MSADataset = msa_dataset
        self.batch_collator: BatchCollator = batch_collator
        self.encoder: TokenEncoder = encoder
        self.chance_flip: float = chance_flip
        self.device: torch.device = device

    @overrides
    def records_lengths(self) -> list[int]:
        return self.index_df["seq_len"].tolist()

    @overrides
    def __len__(self) -> int:
        return len(self.index_df)

    @overrides
    def __getitem__(self, idx: int) -> DataPoint:
        if idx < 0:
            idx += len(self)

        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds for dataset of length {len(self)}")

        row = self.index_df.iloc[idx]
        target_id = row['target_id']

        input = pd.read_csv(os.path.join(self.root_path, f"{target_id}.in"), sep=",").iloc[0]
        label: pd.DataFrame = pd.read_csv(os.path.join(self.root_path, f"{target_id}.gt"), sep=",")

        should_reverse: bool = np.random.rand() < self.chance_flip
        sequence = self.encoder.encode(np.array(list(input['sequence']), dtype='U1'))
        if should_reverse:
            sequence = sequence[::-1]
        sequence = torch.tensor(sequence, dtype=torch.int32, device=self.device)

        has_msa: bool = False
        msa, msa_profiles = self.msa_dataset.get_msa(
            target_id,
            has_msa=has_msa,
            sequence=input['sequence'],
            transform=lambda x: x[:, ::-1] if should_reverse else x
        )

        label.drop(columns=['ID', 'target_id'], inplace=True)
        label.sort_values(by='resid', inplace=True)

        ground_truth: torch.Tensor = torch.zeros([len(sequence), 3], dtype=torch.float32)
        for i, coord in enumerate(['x_1', 'y_1', 'z_1']):
            ground_truth[:, i] = torch.tensor(label[coord].values, dtype=torch.float32)

        result: DataPoint = {
            "target_id": target_id,
            "sequence": sequence,
            "sequence_mask": torch.ones_like(sequence, dtype=torch.bool, device=self.device),
            "has_msa": torch.tensor(has_msa, dtype=torch.bool, device=self.device),
            "msa": torch.tensor(msa, dtype=torch.int32, device=self.device),
            "msa_profiles": torch.tensor(msa_profiles, dtype=torch.float32, device=self.device),
            "num_product_sequences": torch.tensor(0, dtype=torch.int32, device=self.device),
            "product_sequences": None,
            "ground_truth": ground_truth,
            "num_ground_truths": torch.tensor(1, dtype=torch.int32, device=self.device),
            "is_synthetic": torch.tensor(True, dtype=torch.bool, device=self.device),
        }

        return result

    @overrides
    def collate_fn(self, batch: list[DataPoint]) -> DataBatch:
        return self.batch_collator(batch)


if __name__ == "__main__":
    root_path_ = r"E:\Raw Datasets\UW Synthetic RNA structures\converted"
    index_filepath_ = os.path.join(root_path_, "index.csv")

    from data.sequence_padder import SequencePadder
    from data.token_library import TokenLibrary
    token_lib_ = TokenLibrary()
    encoder_ = TokenEncoder(
        tokens=np.array(token_lib_.all_tokens, dtype='U1'),
        map_token_to_id=token_lib_.map_token_to_id,
        missing_token_id=token_lib_.missing_residue_token_id,
    )
    seq_padder_ = SequencePadder(pad_token_id=token_lib_.pad_token_id)
    batch_collator_ = BatchCollator(sequence_padder=seq_padder_)
    dataset_ = UVSyntheticDataset(root_path_, index_filepath_, batch_collator_, encoder_, device=torch.device("cpu"))

    print(f"Dataset length: {len(dataset_)}")
    for i_, record_ in enumerate(dataset_):
        if i_ % 50_000 == 0:
            print("Id: ", record_['target_id'],
                  "Ground truth", record_['ground_truth'])

