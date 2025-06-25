import pytest
import torch

from loss.utils.kabsch_algorithm import kabsch_algorithm


def calc_mse(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (
        torch.square(a - b)
        .sum(dim=-1)
        .masked_fill(~mask, 0.0)
        .sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
    )

def random_rotation_matrix(batch_sizes, device=None, dtype=torch.float32):
    # Uniform quaternion method (Shoemake)
    u1 = torch.rand(batch_sizes, device=device, dtype=dtype)
    u2 = torch.rand(batch_sizes, device=device, dtype=dtype)
    u3 = torch.rand(batch_sizes, device=device, dtype=dtype)
    qx = torch.sqrt(1 - u1) * torch.sin(2 * torch.pi * u2)
    qy = torch.sqrt(1 - u1) * torch.cos(2 * torch.pi * u2)
    qz = torch.sqrt(u1)     * torch.sin(2 * torch.pi * u3)
    qw = torch.sqrt(u1)     * torch.cos(2 * torch.pi * u3)
    # build rotation matrix from (qw, qx, qy, qz)
    qx2, qy2, qz2 = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    xw, yw, zw = qx*qw, qy*qw, qz*qw
    R = torch.stack([
        torch.stack([1 - 2*(qy2 + qz2),     2*(xy - zw),         2*(xz + yw)], dim=-1),
        torch.stack([    2*(xy + zw),   1 - 2*(qx2 + qz2),       2*(yz - xw)], dim=-1),
        torch.stack([    2*(xz - yw),       2*(yz + xw),     1 - 2*(qx2 + qy2)], dim=-1),
    ], dim=-2).to(device)
    return R  # shape ($, 3,3), det ≈ +1


@pytest.mark.parametrize("points_p, points_q, sequence_mask", [
    (torch.randn(shape), torch.randn(shape), torch.randn(shape[:-1]) > 0.5)
    for shape in [(5, 10, 3), (2, 20, 3), (3, 15, 3)] * 50
])
def test_goal_to_minimize_mse(
        points_p: torch.Tensor, points_q: torch.Tensor, sequence_mask: torch.Tensor
):
    aligned_p, rotation_matrix, translation_vector = kabsch_algorithm(points_p, points_q, sequence_mask)
    aligned_p = aligned_p.masked_fill(~sequence_mask[..., None], 0.0)
    points_q = points_q.masked_fill(~sequence_mask[..., None], 0.0)

    assert aligned_p.shape == points_p.shape == points_q.shape
    assert rotation_matrix.shape == points_p.shape[:-2] + 2 * (points_p.shape[-1],)
    assert translation_vector.shape == points_p.shape[:-2] + (points_p.shape[-1],)

    mse_real = calc_mse(aligned_p, points_q, sequence_mask)

    points_a = (
        random_rotation_matrix(rotation_matrix.shape[:-2]).unsqueeze(-3) @ points_p.transpose(-2, -1)
    ).transpose(-2, -1) + torch.randn_like(translation_vector).unsqueeze(-2)
    points_a = points_a.masked_fill(~sequence_mask[..., None], 0.0)
    mse_random = calc_mse(points_a, points_q, sequence_mask)

    assert (mse_real <= mse_random).all()


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
def test_correctness(
        points_p: torch.Tensor, points_q: torch.Tensor, target_aligned_points_p: torch.Tensor,
        target_rotation_matrix: torch.Tensor, target_translation_vector: torch.Tensor,
):
    (predicted_aligned_points_p,
     predicted_rotation_matrix,
     predicted_translation_vector) = kabsch_algorithm(points_p, points_q)

    assert torch.allclose(predicted_aligned_points_p, target_aligned_points_p, atol=1e-4)
    assert torch.allclose(predicted_rotation_matrix, target_rotation_matrix, atol=1e-4)
    assert torch.allclose(predicted_translation_vector, target_translation_vector, atol=1e-4)
