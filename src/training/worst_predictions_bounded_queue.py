import heapq

import torch

from data.typing import DataBatch


class WorstPredictionsBoundedQueue:
    def __init__(self, top_k: int = 16, ):
        self.top_k: int = top_k
        self.worst_heap: list[tuple[float, dict]] = []  # stores (loss, data_dict)

    def push_batch(self, data: DataBatch, loss: tuple[torch.Tensor, ...]):
        (rmse_loss, cross_distance_loss, folding_angles_loss, probability_loss) = loss
        L = data["sequence_mask"].sum(dim=-1)
        for i in range(len(data["sequence"])):
            data = {
                "sequence": image.detach().cpu(),
                'ground truth': ground_tr,
                'predicted': predicted_label,
                'card_value_loss': card_value_loss.item(),
                'card_suit_loss': card_suit_loss.item(),
                'loss': final_loss.item(),
            }
            # Use final_loss as the priority
            entry = (data['final_loss'], data)

            if len(self.worst_heap) < self.top_k:
                heapq.heappush(self.worst_heap, entry)
            else:
                heapq.heappushpop(self.worst_heap, entry)

    def clear(self):
        self.worst_heap.clear()