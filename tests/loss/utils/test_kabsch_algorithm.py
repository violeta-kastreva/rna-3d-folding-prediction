import pytest
import torch

from loss.utils.kabsch_algorithm import kabsch_algorithm


def calc_mse(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (
        torch
        .mean(torch.square(a - b)
              .sum(dim=-1)
              .masked_fill(~mask, 0.0),
              dim=-1,
              keepdim=False)
    )


@pytest.mark.parametrize("points_p, points_q, sequence_mask", [
    (torch.randn(shape), torch.randn(shape), torch.randn(shape[:-1]) > 0.5)
    for shape in [(5, 10, 3), (2, 20, 3), (3, 15, 3)]
])
def test_kabsch_algorithm_goal_to_minimize_mse(
        points_p: torch.Tensor, points_q: torch.Tensor, sequence_mask: torch.Tensor
):
    aligned_p, rotation_matrix, translation_vector = kabsch_algorithm(points_p, points_q, sequence_mask)

    assert aligned_p.shape == points_p.shape == points_q.shape
    assert rotation_matrix.shape == points_p.shape[:-2] + 2 * (points_p.shape[-1],)
    assert translation_vector.shape == points_p.shape[:-2] + (points_p.shape[-1],)

    mse_real = calc_mse(aligned_p, points_q, sequence_mask)

    points_a = torch.randn_like(points_p)
    mse_random = calc_mse(points_a, points_q, sequence_mask)

    assert (mse_real <= mse_random).all()

#TODO: FAILED TESTS

@pytest.mark.parametrize(
    "points_p, points_q, target_aligned_points_p, target_rotation_matrix, target_translation_vector", [
    (torch.tensor([[-1., -1.,  0.],
                   [-2.,  0.,  0.],
                   [ 0., -1., -2.],
                   [ 0.,  0.,  0.]]),
     torch.tensor([[-1., -1., -1.],
                   [-1.,  0., -3.],
                   [-2., -1.,  0.],
                   [-1.,  1., -2.]]),
     torch.tensor([[-1.7274, -0.4546, -2.0411],
                   [-0.4630, -0.5150, -2.6717],
                   [-1.4424, -0.8025,  0.1493],
                   [-1.3672,  0.7721, -1.4365]]),
     torch.tensor([[-0.4521,  0.8123, -0.3685],
                   [ 0.6435,  0.5832,  0.4958],
                   [ 0.6176, -0.0130, -0.7864]]),
     torch.tensor([-1.3672,  0.7721, -1.4365]),),
])
def test_kabsch_algorithm_correctness(
        points_p: torch.Tensor, points_q: torch.Tensor, target_aligned_points_p: torch.Tensor,
        target_rotation_matrix: torch.Tensor, target_translation_vector: torch.Tensor,
):
    (predicted_aligned_points_p,
     predicted_rotation_matrix,
     predicted_translation_vector) = kabsch_algorithm(points_p, points_q)

    assert torch.allclose(predicted_aligned_points_p, target_aligned_points_p, atol=1e-4)
    assert torch.allclose(predicted_rotation_matrix, target_rotation_matrix, atol=1e-4)
    assert torch.allclose(predicted_translation_vector, target_translation_vector, atol=1e-4)
