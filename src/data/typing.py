from typing import Literal, Any, Union

DataPointKey = Literal[
    "target_id",
    "sequence",
    "sequence_mask",
    "has_msa",
    "msa",
    "msa_profiles",
    "num_product_sequences",
    "product_sequences",
    "ground_truth",
    "num_ground_truths",
    "is_synthetic",
]

DataPoint = dict[DataPointKey, Any]

DataBatchKey = Union[Literal["product_sequences_indices"], DataPointKey]
DataBatch = dict[DataBatchKey, Any]
