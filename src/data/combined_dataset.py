from itertools import accumulate

from overrides import overrides

from data.rna_dataset_base import RNADatasetBase
from data.typing import DataPoint, DataBatch


class CombinedDataset(RNADatasetBase):
    def __init__(self, datasets: list[RNADatasetBase]):
        self.datasets: list[RNADatasetBase] = datasets

    @overrides
    def records_lengths(self) -> list[int]:
        return [x for dataset in self.datasets for x in dataset.records_lengths()]

    def cutoffs(self) -> list[int]:
        return list(accumulate(map(len, self.datasets)))[:-1]

    @overrides
    def __len__(self) -> int:
        return sum(len(dataset) for dataset in self.datasets)

    @overrides
    def __getitem__(self, idx: int) -> DataPoint:
        if idx < 0:
            idx += len(self)

        for dataset in self.datasets:
            if idx < len(dataset):
                return dataset[idx]
            idx -= len(dataset)

        raise IndexError("Index out of range")

    @overrides
    def collate_fn(self, batch: list[DataPoint]) -> DataBatch:
        return self.datasets[0].collate_fn(batch)
