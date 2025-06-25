import torch

from data.sequence_padder import SequencePadder
from data.typing import DataPointKey, DataPoint, DataBatch, DataBatchKey


class BatchCollator:
    NON_PADDED_KEYS: list[DataPointKey] = ["has_msa", "num_product_sequences", "num_ground_truths", "is_synthetic"]

    def __init__(self, sequence_padder: SequencePadder):
        self.sequence_padder: SequencePadder = sequence_padder

    def __call__(self, batch: list[DataPoint]) -> DataBatch:
        device: torch.device = batch[0]["ground_truth"].device
        collated_data: DataBatch = dict()
        collated_data["target_id"] = [item["target_id"] for item in batch]

        collated_data["sequence"], collated_data["sequence_mask"] = self.sequence_padder.pad(
            [item["sequence"] for item in batch], do_calculate_mask=True, device=device,
        )

        for key in self.NON_PADDED_KEYS:
            collated_data[key] = torch.tensor([item[key] for item in batch], dtype=batch[0][key].dtype, device=device)

        key: DataPointKey
        for key in ("msa", "msa_profiles"):
            collated_data[key] = self.sequence_padder.pad([
                item[key]
                if item[key] is not None
                else torch.empty((0, *batch[0][key].shape[1:]), dtype=batch[0][key].dtype)
                for item in batch
            ],
                pad_value=0.0 if key == "msa_profiles" else None
            )
        # msa: Shape: (N, max_sequence_length, num_representatives)
        # msa_profiles: Shape: (N, max_sequence_length, profile_length)

        max_num_ground_truth: int = max(collated_data["num_ground_truths"].tolist())
        collated_data["ground_truth"] = [
            torch.full(
                (item["ground_truth"].shape[0], max_num_ground_truth, *item["ground_truth"].shape[2:]),
                fill_value=0.0,
                dtype=batch[0]["ground_truth"].dtype,
                device=device,
            )
            for item in batch
        ]
        for item, padded_item in zip(batch, collated_data["ground_truth"]):
            padded_item[:, :item["num_ground_truths"], :] = item["ground_truth"]

        collated_data["ground_truth"] = self.sequence_padder.pad(
            collated_data["ground_truth"], pad_value=0.0
        )[0]  # Shape: (N, max_sequence_length, max_num_ground_truth, 3)

        collated_data["product_sequences"] = [
            sequence
            for item in batch if item["product_sequences"] is not None
            for sequence in item["product_sequences"]
        ]
        collated_data["product_sequences_indices"] = torch.tensor([
            [i, j]
            for i, item in enumerate(batch) if item["product_sequences"] is not None
            for j in range(item["num_product_sequences"])
        ], dtype=torch.int16, device=device).T  # Shape: (2, num_product_sequences)

        return collated_data
