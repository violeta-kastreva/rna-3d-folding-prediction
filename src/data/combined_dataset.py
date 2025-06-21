from typing import Protocol, TypeVar, runtime_checkable
from itertools import accumulate

from torch.utils.data import Dataset

from data.utils import collate_fn


T_co = TypeVar('T_co', covariant=True)


@runtime_checkable
class SequencesDataset(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> T_co: ...

    def records_lengths(self) -> list[int]: ...


class CombinedDataset(Dataset):
    def __init__(self, datasets: list[SequencesDataset]):
        self.datasets: list[SequencesDataset] = datasets

    def records_lengths(self) -> list[int]:
        return [x for dataset in self.datasets for x in dataset.records_lengths()]

    def cutoffs(self) -> list[int]:
        return list(accumulate(map(len, self.datasets)))[:-1]

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def __getitem__(self, idx):
        if idx < 0:
            idx += len(self)

        for dataset in self.datasets:
            if idx < len(dataset):
                return dataset[idx]
            idx -= len(dataset)

        raise IndexError("Index out of range")

    @staticmethod
    def collate_fn(batch):
        return collate_fn(batch)
