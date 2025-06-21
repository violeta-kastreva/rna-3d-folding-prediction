from typing import Any

import torch
TENSOR_KEYS: list[str] = ["ground_truth", "is_synthetic"]


def collate_fn(batch):
    # Assuming each item in batch is a dictionary with the same keys
    collated_data: dict[str, Any] = {
        key: [item[key] for item in batch]
        for key in batch[0].keys()
        if key not in TENSOR_KEYS
    }

    for key in TENSOR_KEYS:
        collated_data[key] = torch.stack([item[key] for item in batch], dim=0).to(device=batch[0].device)

    return collated_data
