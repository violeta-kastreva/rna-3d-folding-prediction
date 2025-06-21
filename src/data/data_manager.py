import string
from functools import cached_property
from dataclasses import dataclass
from typing import Optional

from torch.utils.data import DataLoader

from data.batch_samplers.train_batch_sampler import TrainBatchSampler
from data.combined_dataset import CombinedDataset
from data.experimental_dataset import ExperimentalDataset
from data.uv_synthetic_dataset import UVSyntheticDataset


@dataclass
class DataManagerConfig:
    train_sequence_files: list[str]
    train_label_files: list[str]

    val_sequence_files: list[str]
    val_label_files: list[str]

    test_sequence_files: list[str]

    msa_folders: list[str]

    synthetic_data_root_path: str
    synthetic_data_index_filepath: str

    train_batch_size: int = 4
    val_batch_size: Optional[int] = None
    test_batch_size: Optional[int] = None

    train_num_workers: int = 4
    val_num_workers: int = 2
    test_num_workers: int = 2

    prefetch_factor: int = 2

    train_shuffle_radius: int = 50

    def __post_init__(self):
        if self.val_batch_size is None:
            self.val_batch_size = self.train_batch_size
        if self.test_batch_size is None:
            self.test_batch_size = self.train_batch_size


class DataManager:
    def __init__(self, config: DataManagerConfig):
        self.config: DataManagerConfig = config

    @cached_property
    def train_dataset(self) -> CombinedDataset:
        return CombinedDataset(
            datasets=[
                ExperimentalDataset(
                    self.config.train_sequence_files,
                    self.config.train_label_files,
                    self.config.msa_folders
                ),
                UVSyntheticDataset(
                    self.config.synthetic_data_root_path,
                    self.config.synthetic_data_index_filepath
                ),
            ],
        )

    @cached_property
    def validation_dataset(self) -> ExperimentalDataset:
        return ExperimentalDataset(
            self.config.val_sequence_files,
            self.config.val_label_files,
            self.config.msa_folders,
        )

    @cached_property
    def test_dataset(self) -> ExperimentalDataset:
        return ExperimentalDataset(
            self.config.test_sequence_files,
            [],
            self.config.msa_folders,
            has_ground_truth=False,
        )

    @cached_property
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            num_workers=self.config.train_num_workers,
            batch_sampler=TrainBatchSampler(
                dataset=self.train_dataset,
                batch_size_base=self.config.train_batch_size,
                shuffle_radius=self.config.train_shuffle_radius,
            ),
            prefetch_factor=self.config.prefetch_factor,
            collate_fn=self.train_dataset.__class__.collate_fn,
        )

    @cached_property
    def validation_dataloader(self) -> DataLoader:
        return DataLoader(
            self.validation_dataset,
            num_workers=self.config.val_num_workers,
            batch_size=self.config.val_batch_size,
            prefetch_factor=self.config.prefetch_factor,
            collate_fn=self.validation_dataset.__class__.collate_fn,
        )

    @cached_property
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            num_workers=self.config.test_num_workers,
            batch_size=self.config.test_batch_size,
            prefetch_factor=self.config.prefetch_factor,
            collate_fn=self.test_dataset.__class__.collate_fn,
        )

    @cached_property
    def all_tokens(self) -> list[str]:
        return list(string.ascii_uppercase) + ["-", "?"]

    @cached_property
    def map_token_to_id(self) -> dict[str, int]:
        return {token: idx for idx, token in enumerate(self.all_tokens)}

    @cached_property
    def pad_token_id(self) -> int:
        return len(self.all_tokens) - 1
