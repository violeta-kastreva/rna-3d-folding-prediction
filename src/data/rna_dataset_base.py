from abc import ABC, abstractmethod
from torch.utils.data import Dataset

from data.typing import DataPoint, DataBatch


class RNADatasetBase(ABC, Dataset):
    @abstractmethod
    def records_lengths(self) -> list[int]:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __getitem__(self, idx: int) -> DataPoint:
        ...

    @abstractmethod
    def collate_fn(self, batch: list[DataPoint]) -> DataBatch:
        ...
