"""Pure-PyTorch multiresolution hash encoding and Instant-NGP INR.

The encoding reproduces tiny-cuda-nn's ``HashGrid``: level scales, the
half-cell offset that staggers levels, table sizes, and the hash function all
match the reference, so a PyTorch model sees the same features for the same
table contents. Only the storage layout of the concatenated table differs.
"""

import math
from collections.abc import Callable
from itertools import product

import numpy as np
import torch
from torch import Tensor, nn

from .mlp import MLP

__all__ = ["HashEncoding", "InstantNGP"]

# tiny-cuda-nn's coherent prime hash factors. The first coordinate is left
# unscaled (factor 1) for cache coherence.
_PRIMES = (1, 2654435761, 805459861)


def _level_scales(
    num_levels: int, base_resolution: int, max_resolution: int
) -> list[float]:
    """Per-level grid scales, computed in float32 like instant-ngp/tiny-cuda-nn.

    The ceiling of a scale sets the level's resolution, so matching the
    reference's float32 rounding keeps table sizes identical.
    """
    f32 = np.float32
    ratio = f32(max_resolution) / f32(base_resolution)
    growth = np.exp(np.log(ratio) / f32(max(num_levels - 1, 1)))
    log2_growth = np.log2(growth)
    return [
        float(f32(2) ** (f32(level) * log2_growth) * f32(base_resolution) - f32(1))
        for level in range(num_levels)
    ]


class HashEncoding(nn.Module):
    """Multiresolution hash encoding (Müller et al., SIGGRAPH 2022).

    Level ``l`` scales unit-cube positions by ``N_min * b**l - 1``, with growth
    ``b = (N_max / N_min) ** (1 / (L - 1))``, and adds a half-cell offset so
    levels are staggered (paper, Appendix A). Its grid has
    ``ceil(scale) + 1`` vertices per axis. Grids whose vertices fit in the
    table are indexed densely; finer grids are hashed. Features of the
    ``2**C`` surrounding vertices are interpolated multilinearly and
    concatenated across levels.

    Args:
        in_features: Number of input coordinates ``C`` (1, 2, or 3).
        num_levels: Number of resolution levels ``L``.
        features_per_level: Feature channels ``F`` per table entry.
        log2_hashmap_size: Base-2 logarithm of the maximum entries ``T`` per level.
        base_resolution: Coarsest resolution ``N_min``.
        max_resolution: Finest resolution ``N_max`` over the ``[-1, 1]`` domain;
            the reference uses 2048 for NeRF and SDFs, ``max(H, W) / 2`` for
            images, and the voxel resolution for volumes.

    Shape:
        - Input: :math:`(*, C)` in ``[-1, 1]``.
        - Output: :math:`(*, L F)`.

    The learnable ``embeddings`` concatenate all levels and start uniformly
    in ``[-1e-4, 1e-4]``.
    """

    def __init__(
        self,
        in_features: int,
        num_levels: int = 16,
        features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        max_resolution: int = 2048,
    ) -> None:
        super().__init__()
        if not 1 <= in_features <= len(_PRIMES):
            raise ValueError(f"in_features must be in [1, {len(_PRIMES)}]")
        self.in_features = in_features
        self.num_levels = num_levels
        self.features_per_level = features_per_level

        # Python scalars avoid device synchronization inside the level loop.
        self.scales = _level_scales(num_levels, base_resolution, max_resolution)
        self.resolutions = [math.ceil(scale) + 1 for scale in self.scales]
        self.table_sizes = [
            -(-min(resolution**in_features, 2**log2_hashmap_size) // 8) * 8
            for resolution in self.resolutions
        ]
        self.offsets = [0]
        for size in self.table_sizes[:-1]:
            self.offsets.append(self.offsets[-1] + size)

        self.register_buffer(
            "corners",
            torch.tensor(list(product((0, 1), repeat=in_features)), dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "primes",
            torch.tensor(_PRIMES[:in_features], dtype=torch.long),
            persistent=False,
        )
        self.embeddings = nn.Parameter(
            torch.empty(sum(self.table_sizes), features_per_level).uniform_(-1e-4, 1e-4)
        )

    @property
    def out_features(self) -> int:
        """Number of output features, ``num_levels * features_per_level``."""
        return self.num_levels * self.features_per_level

    def _index(self, vertices: Tensor, resolution: int, table_size: int) -> Tensor:
        """Table index of integer grid vertices ``(..., C)`` at one level."""
        if table_size < resolution**self.in_features:
            # Spatial hash; the reference's uint32 wraparound only affects bits
            # above log2(table_size), so int64 arithmetic gives the same index.
            hashed = vertices * self.primes
            index = hashed[..., 0]
            for dim in range(1, self.in_features):
                index = index ^ hashed[..., dim]
        else:
            # Dense stride index with the first coordinate varying fastest.
            index = vertices[..., -1]
            for dim in range(self.in_features - 2, -1, -1):
                index = index * resolution + vertices[..., dim]
        return index % table_size

    def forward(self, x: Tensor) -> Tensor:
        batch_shape = x.shape[:-1]
        unit_x = (x.reshape(-1, self.in_features) + 1) / 2

        features: list[Tensor] = []
        for scale, resolution, table_size, offset in zip(
            self.scales, self.resolutions, self.table_sizes, self.offsets
        ):
            position = unit_x * scale + 0.5
            grid = position.floor()
            fraction = (position - grid).unsqueeze(1)  # (N, 1, C)
            vertices = grid.long().unsqueeze(1) + self.corners  # (N, 2^C, C)
            index = self._index(vertices, resolution, table_size)
            corner_features = self.embeddings[offset + index]  # (N, 2^C, F)
            weights = torch.where(self.corners.bool(), fraction, 1 - fraction).prod(-1)
            features.append((weights.unsqueeze(-1) * corner_features).sum(1))

        out = torch.cat(features, dim=-1)
        return out.reshape(*batch_shape, self.out_features)


class InstantNGP(nn.Module):
    r"""Instant neural graphics primitive: hash encoding and a bias-free ReLU MLP.

    Mirrors tiny-cuda-nn's ``HashGrid`` followed by ``FullyFusedMLP``, which
    has no biases. The reference uses one 64-wide hidden layer for the NeRF
    density network and two for images and SDFs. See :class:`HashEncoding`
    for the encoding parameters.

    Args:
        in_features: Number of input coordinates (1, 2, or 3).
        out_features: Number of output channels.
        num_levels: Number of encoding levels.
        features_per_level: Feature channels per table entry.
        log2_hashmap_size: Base-2 logarithm of the maximum entries per level.
        base_resolution: Coarsest grid resolution.
        max_resolution: Finest grid resolution.
        hidden_features: Decoder hidden width.
        hidden_layers: Number of decoder hidden layers; at least one.
        output_activation: Optional callable applied to the output.

    Shape:
        - Input: :math:`(*, \text{in\_features})` in ``[-1, 1]``.
        - Output: :math:`(*, \text{out\_features})`.

    Train with Adam ``betas=(0.9, 0.99)`` and ``eps=1e-15`` as in the reference;
    the tiny epsilon matters for rarely updated table entries.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_levels: int = 16,
        features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        max_resolution: int = 2048,
        hidden_features: int = 64,
        hidden_layers: int = 1,
        output_activation: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.output_activation = (
            output_activation if output_activation is not None else nn.Identity()
        )

        self.encoding = HashEncoding(
            in_features,
            num_levels=num_levels,
            features_per_level=features_per_level,
            log2_hashmap_size=log2_hashmap_size,
            base_resolution=base_resolution,
            max_resolution=max_resolution,
        )
        self.decoder = MLP(
            self.encoding.out_features,
            out_features,
            hidden_features,
            hidden_layers,
            bias=False,  # hidden layers
        )
        # FullyFusedMLP has no output bias either.
        output_layer = self.decoder.layers[-1]
        self.decoder.layers[-1] = nn.Linear(
            output_layer.in_features, out_features, bias=False
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.output_activation(self.decoder(self.encoding(x)))
