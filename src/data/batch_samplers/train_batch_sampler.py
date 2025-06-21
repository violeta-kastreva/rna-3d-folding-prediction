from typing import Iterator
import random
from torch.utils.data import Sampler
import torch

from data.combined_dataset import CombinedDataset


class TrainBatchSampler(Sampler[list[int]]):
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
            self.chunk(
                self._constrained_shuffle(ordered_indices, self.shuffle_radius),
                len(self)
            )
            for i, ordered_indices in enumerate((self.dataset1_ordered_indices,
                                                 self.dataset2_ordered_indices))
        )))

        for batch1, batch2 in random.sample(batches, len(batches)):
            yield batch1 + [val + self.dataset1_len for val in batch2]

    @classmethod
    def _constrained_shuffle(cls, lst: list, k: int) -> list:
        n = len(lst)
        result = [None] * n
        available_indices = list(range(n))

        for i in range(n):
            # Determine the range of indices to select from
            start = max(0, i - k)
            end = min(i + k, n - 1)

            # Find available indices within the allowed range
            candidates = [idx for idx in available_indices if start <= idx <= end]

            if not candidates:
                # If no candidates are available in the desired range, expand the range
                # This handles cases where K is too small to complete the shuffle
                candidates = available_indices

            # Randomly select an index from the candidates
            chosen_idx = random.choice(candidates)
            result[i] = lst[chosen_idx]
            available_indices.remove(chosen_idx)

        return result

    @classmethod
    def chunk(cls, lst: list, n: int) -> list[torch.Tensor]:
        if len(lst) < n:
            result = []
            for i in range((n + len(lst) - 1) // len(lst)):
                result.append(random.sample(lst, len(lst)))

            return result[:n]

        chunk_size = (len(lst)) // n

        rem = (len(lst)) % n
        result = []
        for i in range(rem):
            result.append(lst[i * (chunk_size + 1):(i + 1) * (chunk_size + 1)])
        curr = rem * (chunk_size + 1)
        for i in range(n - rem):
            result.append(lst[curr + i * chunk_size:(i + 1) * chunk_size])
        return result
