from typing import Optional

import torch


def kabsch_algorithm(
        p: torch.Tensor,  # Shape: ($, N, 3)
        q: torch.Tensor,  # Shape: (#, N, 3)
        sequence_mask: Optional[torch.Tensor] = None,  # Shape: ($, N) or None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Step 1: Compute centroids
    centroid_p = p.mean(dim=-2, keepdim=True)  # Shape: ($, 1, 3)
    centroid_q = q.mean(dim=-2, keepdim=True)  # Shape: (#, 1, 3)

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
    rot = v_t.transpose(-2, -1) @ u.transpose(-2, -1)  # Shape: ($, 3, 3)

    # Ensure a proper rotation (no reflection)
    v_t[..., 2, :] = torch.where(
        (torch.linalg.det(rot) < 0.0)[..., None],  # Shape: ($, 1)
        -v_t[..., 2, :],  # Shape: ($, 3)
        v_t[..., 2, :]   # Shape: ($, 3)
    )
    rot = v_t.transpose(-2, -1) @ u.transpose(-2, -1)  # Shape: ($, 3, 3)

    # Step 6: Compute translation vector
    trans = centroid_q - (rot[..., None, :, :] @ centroid_p[..., None]).squeeze(dim=-1)  # Shape: ($, 1, 3)

    # Step 7: Apply transformation
    p_aligned = (rot @ p.transpose(-2, -1)).transpose(-2, -1) + trans  # Shape: ($, N, 3)

    trans = trans.squeeze(dim=-2)  # Shape: ($, 3)
    return p_aligned, rot, trans
