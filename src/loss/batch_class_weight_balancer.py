import torch


class BatchBinaryClassWeightBalancer:
    @classmethod
    def get_class_weight(cls, batch_record_classes: torch.Tensor) -> tuple[float, float]:
        """
        Returns the weight for a each class in a batch of binary classification records.
        The weights are calculated as though records from the smaller class are repeated to balance the classes.

        Args:
            batch_record_classes (torch.Tensor): A tensor of shape (batch_size, 1) containing binary class labels
                (0 or 1) for each record in the batch.
        Returns:
            tuple[float, float]: A tuple containing the weight for the positive class (1) and the negative class (0).
                The weights are calculated as follows:
                - Weight for the positive class = 1 / (2 * number of positive records)
                - Weight for the negative class = 1 / (2 * number of negative records)
        """
        num_true: int = batch_record_classes.sum(dim=-1, dtype=torch.int16).item()
        num_false: int = batch_record_classes.shape[-1] - num_true

        if num_false == 0:
            return 1.0, 0.0
        if num_true == 0:
            return 0.0, 1.0

        true_coefficient: float = 1.0 / (2 * num_true)
        false_coefficient: float = 1.0 / (2 * num_false)

        return true_coefficient, false_coefficient
