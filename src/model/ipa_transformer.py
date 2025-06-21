# add an IPA Transformer - iteratively updating rotations and translations
from typing import Optional

from torch import nn, einsum
import torch

from model.invariant_point_attention import InvariantPointAttention
from model.utils import exists
from model.pytorch3d.transforms import quaternion_multiply, quaternion_to_matrix

# one transformer block based on IPA


def feed_forward(dim: int, hidden_layer_mult_factor: int = 1., num_layers: int = 2, act=nn.ReLU) -> nn.Sequential:
    layers = []
    dim_hidden = dim * hidden_layer_mult_factor

    for ind in range(num_layers):
        is_first = ind == 0
        is_last = ind == (num_layers - 1)
        dim_in = dim if is_first else dim_hidden
        dim_out = dim if is_last else dim_hidden

        layers.append(nn.Linear(dim_in, dim_out))

        if is_last:
            continue

        layers.append(act())

    return nn.Sequential(*layers)


class IPABlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        ff_mult: int = 1,
        ff_num_layers: int = 3,  # in the paper, they used 3 layer transition (feedforward) block
        post_norm: bool = True,  # in the paper, they used post-layer norm - offering pre-norm as well
        post_attn_dropout: float = 0.,
        post_ff_dropout: float = 0.,
        **kwargs,
    ):
        super().__init__()
        self.post_norm = post_norm

        self.attn_norm = nn.LayerNorm(dim)
        self.attn = InvariantPointAttention(d_model=dim, **kwargs)
        self.post_attn_dropout = nn.Dropout(post_attn_dropout)

        self.ff_norm = nn.LayerNorm(dim)
        self.ff = feed_forward(dim, hidden_layer_mult_factor=ff_mult, num_layers=ff_num_layers)
        self.post_ff_dropout = nn.Dropout(post_ff_dropout)

    def forward(self, x, **kwargs):
        post_norm = self.post_norm

        attn_input = x if post_norm else self.attn_norm(x)
        x = self.attn(attn_input, **kwargs) + x
        x = self.post_attn_dropout(x)
        x = self.attn_norm(x) if post_norm else x

        ff_input = x if post_norm else self.ff_norm(x)
        x = self.ff(ff_input) + x
        x = self.post_ff_dropout(x)
        x = self.ff_norm(x) if post_norm else x
        return x


class IPATransformer(nn.Module):
    def __init__(
            self,
            *,
            dim: int,
            depth: int,
            num_tokens: Optional[int] = None,
            predict_points: bool = True,
            detach_rotations: bool = True,
            **kwargs,
    ):
        super().__init__()

        self.token_emb = nn.Embedding(num_tokens, dim) if exists(num_tokens) else None

        # Layers
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                IPABlock(dim=dim, **kwargs),
                nn.Linear(dim, 2 * InvariantPointAttention.DIMS),
            ]))

        # Whether to detach rotations or not, for stability during training
        self.detach_rotations: bool = detach_rotations

        # Output
        self.predict_points: bool = predict_points

        if predict_points:
            self.to_points = nn.Linear(dim, InvariantPointAttention.DIMS)

    def forward(
            self,
            single_repr: torch.Tensor,  # Shape: (B, $, L) - single residue representation
            *,
            translations: Optional[torch.Tensor] = None,  # Shape: (B, $, L, 3) - translations
            quaternions: Optional[torch.Tensor] = None,  # Shape: (B, $, L, 4) - quaternions
            pairwise_repr: Optional[torch.Tensor] = None,  # Shape: (B, $$, L, L, d_pair_repr) - pairwise representation
            sequence_mask: Optional[torch.Tensor] = None,  # Shape: (B, $$, L), $ is broadcastable to $
    ):
        x, device = single_repr, single_repr.device

        if exists(self.token_emb):
            x = self.token_emb(x)

        # If no initial quaternions passed in, start from identity
        if not exists(quaternions):
            quaternions = torch.zeros((*single_repr.shape, 4), device=device)  # initial rotations, Shape: (4,)
            quaternions[..., 0] = 1.  # set the first element to 1, rest to 0

        # If not translations passed in, start from identity
        if not exists(translations):
            translations = torch.zeros((*single_repr.shape, 3), device=device)

        results: Optional[dict] = None
        if self.predict_points:
            # If we are predicting points, we need to initialize the results dictionary
            results = {
                "rotations": [],
                "translations": [],
                "points_local": [],
            }

        # Go through the layers and apply invariant point attention and feedforward
        for block, frame_update_proj in self.layers:
            rotations = quaternion_to_matrix(quaternions)

            if self.detach_rotations:
                rotations = rotations.clone().detach()

            x = block(
                x,
                pairwise_repr=pairwise_repr,
                rotations=rotations,
                translations=translations,
                sequence_mask=sequence_mask,
            )

            # Update quaternion and translation
            quaternion_update, translation_update = frame_update_proj(x).chunk(2, dim=-1)  # Shapes: (B, $, L, DIMS)
            quaternion_update = F.pad(quaternion_update, (1, 0), value=1.)  # Add real part 1. to the quaternion update
            quaternion_update = quaternion_update / torch.linalg.norm(quaternion_update, dim=-1, keepdim=True)
            quaternions = quaternion_multiply(quaternions, quaternion_update)
            translations = InvariantPointAttention.transform_from_local_to_global_frame(
                translation_update, rotations, translations,
            )

            if self.predict_points:
                points_local = self.to_points(x)
                results["rotations"].append(rotations)
                results["translations"].append(translations)
                results["points_local"].append(points_local)

        if not self.predict_points:
            return x, translations, quaternions

        points_local = self.to_points(x)
        rotations = quaternion_to_matrix(quaternions)
        points_global = InvariantPointAttention.transform_from_local_to_global_frame(
            points_local, rotations, translations,
        )
        return points_global
