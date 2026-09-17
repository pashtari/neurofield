"""Grid of independent INRs with smooth overlap blending."""

import math
from collections.abc import Sequence
from itertools import product
from typing import Any

import torch
from torch import Tensor, nn

__all__ = ["GridINR"]


class GridINR(nn.Module):
    """Independent coordinate networks on a regular grid of patches.

    Coordinates in ``[-1, 1]`` are mapped to each patch's local ``[-1, 1]``
    domain. Overlapping patches blend their outputs with Hermite smoothstep
    weights. Up to ``2**in_features`` networks contribute to each point.
    Maps ``(..., in_features)`` to ``(..., out_features)``.

    Args:
        inr_class: Module class accepting ``(in_features, out_features,
            **inr_kwargs)``; a separate instance is built per patch.
        in_features: Number of input coordinates.
        out_features: Number of output channels.
        patches_per_axis: Patch count per axis; an int applies to every axis.
        overlap: Fraction of a cell width added on each side of a patch.
            Zero gives hard boundaries; values above 0.5 cause discontinuities.
        **inr_kwargs: Arguments passed to each patch network.

    The ``inrs`` ModuleList uses row-major patch order, with the last
    coordinate axis varying fastest.
    """

    def __init__(
        self,
        inr_class: type[nn.Module],
        in_features: int,
        out_features: int,
        patches_per_axis: int | Sequence[int] = 2,
        overlap: float = 0.25,
        **inr_kwargs: Any,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.overlap = overlap

        if isinstance(patches_per_axis, int):
            patches_per_axis = (patches_per_axis,) * in_features
        if len(patches_per_axis) != in_features:
            raise ValueError(
                f"patches_per_axis must have {in_features} entries, "
                f"got {len(patches_per_axis)}"
            )
        self.patches_per_axis = patches_per_axis

        num_patches = math.prod(patches_per_axis)
        self.inrs = nn.ModuleList(
            [
                inr_class(in_features, out_features, **inr_kwargs)
                for _ in range(num_patches)
            ]
        )

        # Row-major strides map a cell coordinate to its patch network.
        strides: list[int] = []
        stride = 1
        for patch_count in reversed(patches_per_axis):
            strides.append(stride)
            stride *= patch_count
        self.register_buffer(
            "_strides", torch.tensor(list(reversed(strides)), dtype=torch.long)
        )
        self.register_buffer("_grid", torch.tensor(patches_per_axis, dtype=torch.float))
        self.register_buffer(
            "_grid_long", torch.tensor(patches_per_axis, dtype=torch.long)
        )

    @staticmethod
    def _smoothstep(t: Tensor) -> Tensor:
        """Blend from zero to one with zero derivatives at both endpoints."""
        t = t.clamp(0, 1)
        return t * t * (3 - 2 * t)

    def _to_local(self, fraction: Tensor) -> Tensor:
        """Map cell coordinates to [-1, 1], reserving room for overlap."""
        if self.overlap > 0:
            return (fraction + self.overlap) / (1 + 2 * self.overlap) * 2 - 1
        return fraction * 2 - 1

    def _eval_patches(self, patch_indices: Tensor, local_x: Tensor) -> Tensor:
        """Group coordinates by patch to evaluate contiguous slices of each INR."""
        order = patch_indices.argsort()
        sorted_x = local_x[order]
        sorted_patches = patch_indices[order]

        unique_patches, counts = torch.unique_consecutive(
            sorted_patches, return_counts=True
        )

        results: list[Tensor] = []
        offset = 0
        for patch_index, count in zip(unique_patches.tolist(), counts.tolist()):
            results.append(self.inrs[patch_index](sorted_x[offset : offset + count]))
            offset += count

        sorted_out = torch.cat(results, dim=0)

        out = torch.empty_like(sorted_out)
        out[order] = sorted_out
        return out

    def forward(self, x: Tensor) -> Tensor:
        *batch_shape, num_dims = x.shape
        x_flat = x.reshape(-1, num_dims)
        num_points = x_flat.shape[0]

        unit_x = (x_flat + 1) / 2
        position = unit_x * self._grid
        cell = position.long().clamp(min=0)
        cell = torch.min(cell, (self._grid_long - 1).unsqueeze(0))
        fraction = position - cell.float()

        if self.overlap == 0:
            local_x = self._to_local(fraction)
            patch_indices = (cell * self._strides).sum(-1)
            out = self._eval_patches(patch_indices, local_x)
            return out.reshape(*batch_shape, self.out_features)

        # Blend only at interior boundaries; edge patches retain full weight.
        left_mask = (fraction < self.overlap) & (cell > 0)
        right_mask = (fraction > 1 - self.overlap) & (
            cell < self._grid_long.unsqueeze(0) - 1
        )

        axis_weights = torch.ones(num_points, num_dims, device=x.device, dtype=x.dtype)
        axis_weights = torch.where(
            left_mask,
            0.5 + 0.5 * self._smoothstep(fraction / self.overlap),
            axis_weights,
        )
        axis_weights = torch.where(
            right_mask,
            0.5 + 0.5 * self._smoothstep((1 - fraction) / self.overlap),
            axis_weights,
        )

        neighbor = torch.where(left_mask, cell - 1, cell)
        neighbor = torch.where(right_mask, cell + 1, neighbor)

        neighbor_fraction = torch.where(neighbor < cell, fraction + 1, fraction - 1)

        # Dispatch once so each patch processes all of its contributing points.
        all_patch_indices: list[Tensor] = []
        all_local_x: list[Tensor] = []
        all_weights: list[Tensor] = []
        all_point_indices: list[Tensor] = []

        for bits in product([0, 1], repeat=num_dims):
            patch_weights = torch.ones(num_points, device=x.device, dtype=x.dtype)
            for axis, use_neighbor in enumerate(bits):
                weight = axis_weights[:, axis]
                patch_weights = patch_weights * (1 - weight if use_neighbor else weight)

            active = patch_weights > 1e-6
            if not active.any():
                continue

            active_indices = torch.where(active)[0]

            # Each bit selects the primary cell or its neighbor along one axis.
            neighbor_axes = torch.tensor(bits, device=x.device, dtype=torch.bool)
            selected_cells = torch.where(
                neighbor_axes, neighbor[active_indices], cell[active_indices]
            )
            selected_fraction = torch.where(
                neighbor_axes,
                neighbor_fraction[active_indices],
                fraction[active_indices],
            )

            all_patch_indices.append((selected_cells * self._strides).sum(-1))
            all_local_x.append(self._to_local(selected_fraction))
            all_weights.append(patch_weights[active_indices])
            all_point_indices.append(active_indices)

        patch_indices = torch.cat(all_patch_indices)
        local_x = torch.cat(all_local_x)
        patch_outputs = self._eval_patches(patch_indices, local_x)

        # Accumulate contributions in the original point order.
        weights = torch.cat(all_weights).unsqueeze(-1)
        point_indices = torch.cat(all_point_indices)

        out = torch.zeros(num_points, self.out_features, device=x.device, dtype=x.dtype)
        out.scatter_add_(
            0,
            point_indices.unsqueeze(-1).expand_as(patch_outputs),
            weights * patch_outputs,
        )

        return out.reshape(*batch_shape, self.out_features)
