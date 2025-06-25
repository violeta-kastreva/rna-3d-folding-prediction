from typing import Optional

import torch
from torch.onnx.symbolic_opset11 import unsqueeze


def kabsch_algorithm(
        p: torch.Tensor,  # Shape: ($, N, 3)
        q: torch.Tensor,  # Shape: (#, N, 3)
        sequence_mask: Optional[torch.Tensor] = None,  # Shape: ($, N) or None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Kabsch algorithm to align two sets of 3D points (p and q) by computing the optimal rotation and translation.
    :param p: Points to be aligned, shape ($, N, 3) where $ is the batch size and N is the number of points.
    :param q: Points to align to, shape (#, N, 3) where # is the number of points in the target set.
    :param sequence_mask:
    :return:
    """
    # Step 1: Compute centroids
    if sequence_mask is not None:
        mask = sequence_mask[..., None]  # valid positions = True # Shape: ($, N, 1)
        p = p.masked_fill(~mask, 0.0)
        q = q.masked_fill(~mask, 0.0)
        denom = mask.sum(dim=-2, keepdim=True) # Shape: ($, 1, 1)
        # avoid division by zero
        denom = denom.clamp(min=1)

        centroid_p = p.sum(dim=-2, keepdim=True) / denom  # Shape: ($, 1, 3)
        centroid_q = q.sum(dim=-2, keepdim=True) / denom  # Shape: ($, 1, 3)
    else:
        centroid_p = p.mean(dim=-2, keepdim=True)  # Shape: ($, 1, 3)
        centroid_q = q.mean(dim=-2, keepdim=True)  # Shape: ($, 1, 3)

    # Step 2: Center the points
    p_centered = p - centroid_p  # Shape: ($, N, 3)
    q_centered = q - centroid_q  # Shape: ($, N, 3)

    if sequence_mask is not None:
        mask = ~sequence_mask[..., None] # Shape: ($, N, 1)
        p_centered = p_centered.masked_fill(mask, 0.0)
        q_centered = q_centered.masked_fill(mask, 0.0)

    # Step 3: Compute covariance matrix
    covariance = torch.matmul(p_centered.transpose(-2, -1), q_centered)  # Shape: ($, 3, 3)

    # Step 4: Perform SVD
    u, s, v_t = torch.linalg.svd(covariance)  # Shape: u: ($, 3, 3), s: ($, 3), v_t: ($, 3, 3)

    # Step 5: Compute rotation matrix
    # Determine if reflection occurs
    d = torch.sign(torch.linalg.det(v_t.transpose(-2, -1) @ u.transpose(-2, -1)))  # shape: ($,)
    # Prepare diagonal correction B = diag(1,1,d)
    B = (
        torch.eye(3, device=p.device)
        .view(*(1 for _ in covariance.shape[:-2]), 3, 3)
        .repeat((*covariance.shape[:-2], 1, 1))
    )  # Shape: ($, 3, 3)
    B[..., 2, 2] = d

    # Compute proper rotation
    rot = v_t.transpose(-2, -1) @ B @ u.transpose(-2, -1)  # Shape: ($, 3, 3)

    # Step 6: Compute translation vector
    trans = centroid_q - (rot[..., None, :, :] @ centroid_p[..., None]).squeeze(dim=-1)  # Shape: ($, 1, 3)

    # Step 7: Apply transformation
    p_aligned = (rot @ p.transpose(-2, -1)).transpose(-2, -1) + trans  # Shape: ($, N, 3)

    trans = trans.squeeze(dim=-2)  # Shape: ($, 3)
    return p_aligned, rot, trans
