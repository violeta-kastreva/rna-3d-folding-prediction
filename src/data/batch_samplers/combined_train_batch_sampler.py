from typing import Iterator
import random
from torch.utils.data import Sampler
import torch

from data.batch_samplers.train_batch_sampler import TrainBatchSampler
from data.combined_dataset import CombinedDataset


class CombinedTrainBatchSampler(Sampler[list[int]]):
    def __init__(
            self,
            dataset: CombinedDataset,
            batch_size_base: int,
            shuffle_radius: int,
    ) -> None:
        super().__init__()
        self.batch_size: int = batch_size_base
        self.shuffle_radius: int = shuffle_radius

        self.dataset1_len: int = dataset.cutoffs()[0]
        self.dataset2_len: int = len(dataset) - self.dataset1_len
        self.data_len: int = len(dataset)

        self.synthetic_ratio: float = self.dataset2_len / self.data_len

        self.dataset1_ordered_indices: list[int] = torch.argsort(
            torch.tensor(dataset.records_lengths()[:self.dataset1_len], dtype=torch.int16)
        ).tolist()
        self.dataset2_ordered_indices: list[int] = torch.argsort(
            torch.tensor(dataset.records_lengths()[self.dataset1_len:], dtype=torch.int16)
        ).tolist()

    def dataset2_length(self) -> int:
        return round(self.synthetic_ratio * self.data_len)

    def __len__(self) -> int:
        return (self.dataset1_len + self.dataset2_length() + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        batches = list(zip(*(
            TrainBatchSampler.chunk(
                TrainBatchSampler.constrained_shuffle(ordered_indices, self.shuffle_radius),
                len(self)
            )
            for i, ordered_indices in enumerate((self.dataset1_ordered_indices,
                                                 self.dataset2_ordered_indices))
        )))

        for batch1, batch2 in random.sample(batches, len(batches)):
            yield batch1 + [val + self.dataset1_len for val in batch2]

