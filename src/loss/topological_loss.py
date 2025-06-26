from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from data.typing import DataBatch
from loss.batch_class_weight_balancer import BatchBinaryClassWeightBalancer
from loss.utils.kabsch_algorithm import kabsch_algorithm


@dataclass
class LossConfig:
    rmse_weight: float = 0.4
    cross_distance_weight: float = 0.175
    folding_angle_weight: float = 0.175
    probability_weight: float = 0.25
    probability_temperature: float = 1.0
    eps: float = 1e-6


class TopologicalLoss(nn.Module):
    def __init__(
            self,
            rmse_weight: float = 0.4,
            cross_distance_weight: float = 0.175,
            folding_angle_weight: float = 0.175,
            probability_weight: float = 0.25,
            probability_temperature: float = 1.0,
            eps: float = 1e-6,
    ):
        super().__init__()
        self.rmse_weight: float = rmse_weight
        self.cross_distance_weight: float = cross_distance_weight
        self.folding_angle_weight: float = folding_angle_weight
        self.probability_weight: float = probability_weight
        self.cosine_similarity = nn.CosineSimilarity(dim=-1, eps=eps)
        self.tau: float = probability_temperature

    def forward(
            self,
            predicted: tuple[torch.Tensor, torch.Tensor],  # Shape: ((B, $, max_sequence_length, 3), (B, $, max_sequence_length, 1))
            data: DataBatch,  # Shape: {"ground_truth": (B, $, max_num_ground_truth, max_sequence_length, 3) }
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        (predicted_coords, predicted_probs) = predicted
        # output_probs: Shape: (B, $, max_sequence_length, 1)
        predicted = predicted_coords  # Shape: (B, $, max_sequence_length, 3)
        predicted = predicted.unsqueeze(-3)  # Shape: (B, $, 1, max_sequence_length, 3)
        ground_truth = data["ground_truth"].transpose(-2, -3)  # Shape: (B, $, max_num_ground_truth, max_sequence_length, 3)
        sequence_mask = data["sequence_mask"].unsqueeze(-2)  # Shape: (B, $, 1, max_sequence_length)
        ground_truth_mask = data["ground_truth_mask"].transpose(-1, -2)  # Shape: (B, $, max_num_ground_truth, max_sequence_length)
        ground_truth_mask = ground_truth_mask & sequence_mask
        sequence_mask = ground_truth_mask  # Shape: (B, $, max_num_ground_truth, max_sequence_length)
        L_gt = sequence_mask.int().sum(dim=-1)  # Shape: (B, $, max_num_ground_truth)
        # L = sequence_mask.int().sum(dim=-1)  # Shape: (B, $, 1)  # (L_gt <= L).all
        L_gt = L_gt.masked_fill(L_gt < 2, 2)
        L = L_gt
        is_synthetic = data["is_synthetic"]  # Shape: (B, $)

        # predicted_aligned: (B, $, max_num_ground_truth, max_sequence_length, 3)
        # rot: Shape: (B, $, max_num_ground_truth, 3, 3)
        # trans: Shape: (B, $, max_num_ground_truth, 3)
        predicted_aligned, rotations, translations = kabsch_algorithm(predicted, ground_truth, sequence_mask)

        mask_unsqueezed = sequence_mask.unsqueeze(-1)  # Shape: (B, $, 1, max_sequence_length, 1)
        predicted_aligned = torch.where(mask_unsqueezed, predicted_aligned, 0.0)  # Mask out padded values
        ground_truth = torch.where(mask_unsqueezed, ground_truth, 0.0)  # Mask out padded values

        rmse_loss = self.calculate_rmse(ground_truth, predicted_aligned, L)  # Shape: (B, $, max_num_ground_truth)
        cross_distance_loss = self.calculate_cross_distance_loss(
            predicted_aligned,
            ground_truth,
            sequence_mask,
            L,
        )  # Shape: (B, $, max_num_ground_truth)
        folding_angles_loss = self.calculate_folding_angles_loss(
            predicted_aligned,
            ground_truth,
            sequence_mask,
            L,
        )  # Shape: (B, $, max_num_ground_truth)

        probability_loss = self.calculate_probability_loss(
            predicted_probs,
            predicted_aligned,
            ground_truth,
            sequence_mask,
            L,
        )  # Shape: (B, $, max_num_ground_truth)

        loss = (
            self.rmse_weight * rmse_loss +
            self.cross_distance_weight * cross_distance_loss +
            self.folding_angle_weight * folding_angles_loss +
            self.probability_weight * probability_loss
        )  # Shape: (B, $, max_num_ground_truth)

        synthetic_weight, experimental_weight = BatchBinaryClassWeightBalancer.get_class_weight(is_synthetic)
        params = (is_synthetic, (synthetic_weight, experimental_weight))

        loss = self.aggregate_loss(loss, *params)
        agg_rmse_loss = self.aggregate_loss(rmse_loss, *params)
        agg_cross_distance_loss = self.aggregate_loss(cross_distance_loss, *params)
        agg_folding_angles_loss = self.aggregate_loss(folding_angles_loss, *params)
        agg_probability_loss = self.aggregate_loss(probability_loss, *params)

        return (
            loss,
            (agg_rmse_loss, agg_cross_distance_loss, agg_folding_angles_loss, agg_probability_loss),
            (rmse_loss, cross_distance_loss, folding_angles_loss, probability_loss),
        )

    def calculate_rmse(
        self,
        y: torch.Tensor,  # Shape: (B, $, L, 3)
        y_hat: torch.Tensor,  # Shape: (B, $, L, 3)
        L: torch.Tensor,  # Shape: (B, $$)
    ) -> torch.Tensor:  # Shape: (B, $)
        return torch.sqrt(((y - y_hat) ** 2).sum(dim=-1)).sum(dim=-1) / L

    def calculate_cross_distance_loss(
        self,
        y: torch.Tensor,  # Shape: (B, $, max_num_ground_truth, max_sequence_length, 3)
        y_hat: torch.Tensor,  # Shape: (B, $, max_num_ground_truth, max_sequence_length, 3)
        sequence_mask: torch.Tensor,  # Shape: (B, $, 1, max_sequence_length)
        L: torch.Tensor,  # Shape: (B, $, 1)
    ) -> torch.Tensor:  # Shape: (B, $, max_num_ground_truth)
        y_distances = self.calculate_cross_distances(
            y, sequence_mask
        )  # Shape: (B, $, max_num_ground_truth, max_sequence_length, max_sequence_length)
        y_hat_distances = self.calculate_cross_distances(
            y_hat, sequence_mask
        )  # Shape: (B, $, max_num_ground_truth, max_sequence_length, max_sequence_length)

        cross_distance_loss = ((
            (y_distances - y_hat_distances) ** 2)
            .sum(dim=(-1, -2)) / (L * (L - 1))
        )  # Shape: (B, $, max_num_ground_truth)

        return cross_distance_loss

    def calculate_cross_distances(
        self,
        y: torch.Tensor,  # Shape: (B, $, L, 3)
        seq_mask: torch.Tensor,  # Shape: (B, $, L),
    ) -> torch.Tensor:  # Shape: (B, $, L, L)
        distances = torch.linalg.vector_norm(
            y.unsqueeze(-2) - y.unsqueeze(-3),  # Shape: (B, $, L, 1, 3) - (B, $, 1, L, 3) = (B, $, L, L, 3)
            dim=-1
        )  # Shape: (B, $, L, L)
        pair_mask = seq_mask.unsqueeze(-1) & seq_mask.unsqueeze(-2)  # Shape: (B, $, L, 1) - (B, $, 1, L) = (B, $, L, L)
        distances = torch.where(pair_mask, distances, 0.0)

        return distances  # Shape: (B, $, L, L)

    def calculate_folding_angles_loss(
        self,
        y: torch.Tensor,  # Shape: (B, $, max_num_ground_truth, max_sequence_length, 3)
        y_hat: torch.Tensor,  # Shape: (B, $, max_num_ground_truth, max_sequence_length, 3)
        sequence_mask: torch.Tensor,  # Shape: (B, $, 1, max_sequence_length)
        L: torch.Tensor,  # Shape: (B, $, 1)
    ):
        y_folding_angles = self.calculate_cos_of_folding_angles(
            y, sequence_mask
        )  # Shape: (B, $, max_num_ground_truth, max_sequence_length - 2)
        y_hat_folding_angles = self.calculate_cos_of_folding_angles(
            y_hat, sequence_mask
        )  # Shape: (B, $, max_num_ground_truth, max_sequence_length - 2)
        folding_angles_loss = (
                torch.abs(y_folding_angles - y_hat_folding_angles).sum(dim=-1) / L
        )  # Shape: (B, $, max_num_ground_truth)

        return folding_angles_loss

    def calculate_cos_of_folding_angles(
        self,
        y: torch.Tensor,  # Shape: (B, $, L, 3)
        seq_mask: torch.Tensor,  # Shape: (B, $, L)
    ):
        y_prev = y[..., :-2, :]  # Shape: (B, $, L-2, 3)
        y_next = y[..., 2:, :]  # Shape: (B, $, L-2, 3)
        y_curr = y[..., 1:-1, :]  # Shape: (B, $, L-2, 3)

        seq_mask = seq_mask[..., :-2] & seq_mask[..., 2:] & seq_mask[..., 1:-1]  # Shape: (B, $, L-2)

        v1 = y_prev - y_curr  # Shape: (B, $, L-2, 3)
        v2 = y_next - y_curr  # Shape: (B, $, L-2, 3)

        cos_angles = self.cosine_similarity(v1, v2)  # Shape: (B, $, L-2)
        cos_angles = torch.where(seq_mask, cos_angles, 0.0)
        return cos_angles

    def calculate_probability_loss(
        self,
        predicted_probs_logits: torch.Tensor,  # Shape: (B, $, max_sequence_length, 1)
        y: torch.Tensor,  # Shape: (B, $, max_num_ground_truth, max_sequence_length, 3)
        y_hat: torch.Tensor,  # Shape: (B, $, max_num_ground_truth, max_sequence_length, 3)
        sequence_mask: torch.Tensor,  # Shape: (B, $, 1, max_sequence_length)
        L: torch.Tensor,  # Shape: (B, $, 1)
    ) -> torch.Tensor:
        distances = torch.sqrt(((y - y_hat) ** 2).sum(dim=-1))  # Shape: (B, $, max_num_ground_truth, max_sequence_length)
        ground_truth_probs = self.to_prob_exp(distances)  # Shape: (B, $, max_num_ground_truth, max_sequence_length)

        predicted_probs_logits = predicted_probs_logits.squeeze(-1).unsqueeze(-2)  # Shape: (B, $, 1, max_sequence_length)

        predicted_probs_logits = torch.where(sequence_mask, predicted_probs_logits, 0)  # Shape: (B, $, 1, max_sequence_length)
        ground_truth_probs = torch.where(sequence_mask, ground_truth_probs, 0.5)  # Shape: (B, $, max_num_ground_truth, max_sequence_length)

        probs_loss = F.binary_cross_entropy_with_logits(
            predicted_probs_logits.repeat((*(1 for _ in ground_truth_probs.shape[:-2]), ground_truth_probs.shape[-2], 1)),
            ground_truth_probs,
            reduction="none",
        )  # Shape: (B, $, max_num_ground_truth, max_sequence_length)

        probs_loss = probs_loss.sum(dim=-1) / L  # Shape: (B, $, max_num_ground_truth)

        return probs_loss


    def aggregate_loss(
        self,
        loss: torch.Tensor,  # Shape: (B, $, max_num_ground_truth)
        data_point_classes: torch.Tensor,  # Shape: (B, $)
        class_weights: tuple[float, float],
    ) -> torch.Tensor:  # Shape: (,)
        loss = torch.where(
            data_point_classes.unsqueeze(-1),
            loss * class_weights[0],
            loss * class_weights[1],
        )

        loss = loss.mean(dim=tuple(range(-2, -len(loss.shape) -1, -1)))  # Shape: (max_num_ground_truth,)
        loss = loss.min()  # Shape: (,)

        return loss

    def to_prob_exp(self, x: torch.Tensor) -> torch.Tensor:
        # x: non-negative tensor or float
        return torch.exp(-x / self.tau)
