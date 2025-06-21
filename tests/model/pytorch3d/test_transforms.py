import pytest
import torch

from model.pytorch3d.transforms import quaternion_multiply, quaternion_to_matrix


@pytest.mark.parametrize(
    "quaternion1, quaternion2",
    [
        (torch.randn(shape), torch.randn(shape))
        for shape in [(4,), (1, 4), (2, 4)]
        for _ in range(3)
    ]
)
def test_quaternion_multiply(quaternion1, quaternion2):
    product = quaternion_multiply(quaternion1, quaternion2)
    assert product.shape == quaternion1.shape == quaternion2.shape


@pytest.mark.parametrize(
    "quaternion",
    [
        (torch.randn(shape))
        for shape in [(4,), (1, 4), (2, 4)]
    ]
)
def test_quaternion_to_matrix(quaternion):
    matrix = quaternion_to_matrix(quaternion)

    assert matrix.shape == quaternion.shape[:-1] + (3, 3)
