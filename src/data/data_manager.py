from functools import cached_property
from dataclasses import dataclass
from typing import Optional, cast

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from data.batch_collator import BatchCollator
from data.batch_samplers.combined_train_batch_sampler import CombinedTrainBatchSampler
from data.batch_samplers.train_batch_sampler import TrainBatchSampler
from data.combined_dataset import CombinedDataset
from data.experimental_dataset import ExperimentalDataset
from data.msa.msa_dataset import MSAConfig, MSADataset
from data.rna_dataset_base import RNADatasetBase
from data.sequence_padder import SequencePadder
from data.token_encoder import TokenEncoder
from data.token_library import TokenLibrary
from data.uv_synthetic_dataset import UVSyntheticDataset


@dataclass
class DataManagerConfig:
    train_sequence_files: list[str]
    train_label_files: list[str]

    val_sequence_files: list[str]
    val_label_files: list[str]

    test_sequence_files: list[str]
    test_label_files: list[str]

    msa_folders: list[str]
    msa_config: MSAConfig

    combine_synthetic_with_real_data: bool = True
    synthetic_data_root_path: Optional[str] = None
    synthetic_data_index_filepath: Optional[str] = None

    train_batch_size: int = 4
    val_batch_size: Optional[int] = None
    test_batch_size: Optional[int] = None

    train_num_workers: int = 4
    val_num_workers: int = 2
    test_num_workers: int = 2

    prefetch_factor: int = 2

    train_shuffle_radius: int = 50

    chance_flip_sequences: float = 0.5
    chance_use_msa_when_available: float = 0.95

    def __post_init__(self):
        if self.combine_synthetic_with_real_data and (
            self.synthetic_data_root_path is None or
            self.synthetic_data_index_filepath is None
        ):
            raise ValueError(
                "Synthetic data root path and index file path must be provided if "
                "combine_synthetic_with_real_data is True."
            )
        if self.val_batch_size is None:
            self.val_batch_size = self.train_batch_size
        if self.test_batch_size is None:
            self.test_batch_size = self.train_batch_size


class DataManager(TokenLibrary):
    def __init__(self, config: DataManagerConfig, device: torch.device):
        super().__init__()
        self.config: DataManagerConfig = config
        self.device: torch.device = device

    @cached_property
    def token_encoder(self) -> TokenEncoder:
        return TokenEncoder(
            np.array(self.all_tokens),
            self.map_token_to_id,
            self.missing_residue_token_id,
        )

    @cached_property
    def msa_dataset(self) -> MSADataset:
        return MSADataset(
            msa_folders=self.config.msa_folders,
            msa_config=self.config.msa_config,
            residues=self.rna_tokens + [self.missing_residue_token],
            token_encoder=self.token_encoder,
        )

    @cached_property
    def sequence_padder(self) -> SequencePadder:
        return SequencePadder(pad_token_id=self.pad_token_id)

    @cached_property
    def batch_collator(self) -> BatchCollator:
        return BatchCollator(sequence_padder=self.sequence_padder)

    @cached_property
    def train_dataset(self) -> RNADatasetBase:
        experimental_dataset = ExperimentalDataset(
            sequence_files=self.config.train_sequence_files,
            label_files=self.config.train_label_files,
            msa_dataset=self.msa_dataset,
            sequence_padder=self.sequence_padder,
            batch_collator=self.batch_collator,
            chance_flip=self.config.chance_flip_sequences,
            chance_use_msa_when_available=self.config.chance_use_msa_when_available,
            device=self.device,
        )
        if not self.config.combine_synthetic_with_real_data:
            return experimental_dataset

        return CombinedDataset(
            datasets=[
                experimental_dataset,
                UVSyntheticDataset(
                    root_path=self.config.synthetic_data_root_path,
                    index_filepath=self.config.synthetic_data_index_filepath,
                    msa_dataset=self.msa_dataset,
                    batch_collator=self.batch_collator,
                    encoder=self.token_encoder,
                    chance_flip=self.config.chance_flip_sequences,
                    device=self.device,
                ),
            ],
        )

    @cached_property
    def validation_dataset(self) -> RNADatasetBase:
        return ExperimentalDataset(
            sequence_files=self.config.val_sequence_files,
            label_files=self.config.val_label_files,
            msa_dataset=self.msa_dataset,
            sequence_padder=self.sequence_padder,
            batch_collator=self.batch_collator,
            chance_flip=0.0,
            chance_use_msa_when_available=1.0,
            device=self.device,
        )

    @cached_property
    def test_dataset(self) -> RNADatasetBase:
        return ExperimentalDataset(
            sequence_files=self.config.test_sequence_files,
            label_files=self.config.test_label_files,
            msa_dataset=self.msa_dataset,
            sequence_padder=self.sequence_padder,
            batch_collator=self.batch_collator,
            chance_flip=0.0,
            chance_use_msa_when_available=1.0,
            device=self.device,
        )

    @cached_property
    def train_dataloader(self) -> DataLoader:
        batch_sampler: Sampler[list[int]]
        if self.config.combine_synthetic_with_real_data:
            batch_sampler = CombinedTrainBatchSampler(
                dataset=cast(CombinedDataset, self.train_dataset),
                batch_size_base=self.config.train_batch_size,
                shuffle_radius=self.config.train_shuffle_radius,
            )
        else:
            batch_sampler = TrainBatchSampler(
                dataset=self.train_dataset,
                batch_size_base=self.config.train_batch_size,
                shuffle_radius=self.config.train_shuffle_radius,
            )
        return DataLoader(
            self.train_dataset,
            num_workers=self.config.train_num_workers,
            batch_sampler=batch_sampler,
            prefetch_factor=self.config.prefetch_factor,
            collate_fn=self.batch_collator.__call__,
        )

    @cached_property
    def validation_dataloader(self) -> DataLoader:
        return DataLoader(
            self.validation_dataset,
            num_workers=self.config.val_num_workers,
            batch_size=self.config.val_batch_size,
            prefetch_factor=self.config.prefetch_factor,
            collate_fn=self.batch_collator.__call__,
        )

    @cached_property
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            num_workers=self.config.test_num_workers,
            batch_size=self.config.test_batch_size,
            prefetch_factor=self.config.prefetch_factor,
            collate_fn=self.batch_collator.__call__,
        )
