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

    The forward pass handles every level at once, as one batched tensor
    operation per step rather than a loop over levels: the arithmetic is the
    same, so the features are identical, but a pass costs a few dozen kernel
    launches instead of a few hundred, which is what sets the encoding's time
    on the small batches a ray marcher or a training step feeds it.
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

        self.scales = _level_scales(num_levels, base_resolution, max_resolution)
        self.resolutions = [math.ceil(scale) + 1 for scale in self.scales]
        self.table_sizes = [
            -(-min(resolution**in_features, 2**log2_hashmap_size) // 8) * 8
            for resolution in self.resolutions
        ]
        self.offsets = [0]
        for size in self.table_sizes[:-1]:
            self.offsets.append(self.offsets[-1] + size)
        # The per-level constants as tensors, for the batched forward pass. A
        # level whose vertices outnumber its table is hashed, the others are
        # indexed densely.
        levels = {
            "level_scales": torch.tensor(self.scales, dtype=torch.float32),
            "level_resolutions": torch.tensor(self.resolutions, dtype=torch.long),
            "level_table_sizes": torch.tensor(self.table_sizes, dtype=torch.long),
            "level_offsets": torch.tensor(self.offsets, dtype=torch.long),
            "level_hashed": torch.tensor(
                [
                    size < resolution**in_features
                    for resolution, size in zip(self.resolutions, self.table_sizes)
                ]
            ),
        }
        for name, value in levels.items():
            self.register_buffer(name, value, persistent=False)

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

    def forward(self, x: Tensor) -> Tensor:
        batch_shape = x.shape[:-1]
        unit_x = (x.reshape(-1, self.in_features) + 1) / 2

        # Every level at once: (N, L, C) positions, (N, L, 2^C, C) vertices.
        position = unit_x.unsqueeze(1) * self.level_scales.view(1, -1, 1) + 0.5
        grid = position.floor()
        fraction = (position - grid).unsqueeze(2)
        vertices = grid.long().unsqueeze(2) + self.corners

        # Spatial hash; the reference's uint32 wraparound only affects bits
        # above log2(table_size), so int64 arithmetic gives the same index.
        hashed = vertices * self.primes
        hash_index = hashed[..., 0]
        for dim in range(1, self.in_features):
            hash_index = hash_index ^ hashed[..., dim]
        # Dense stride index with the first coordinate varying fastest.
        resolution = self.level_resolutions.view(1, -1, 1)
        dense_index = vertices[..., -1]
        for dim in range(self.in_features - 2, -1, -1):
            dense_index = dense_index * resolution + vertices[..., dim]
        index = torch.where(self.level_hashed.view(1, -1, 1), hash_index, dense_index)
        index = index % self.level_table_sizes.view(1, -1, 1)

        corner_features = self.embeddings[self.level_offsets.view(1, -1, 1) + index]
        weights = torch.where(self.corners.bool(), fraction, 1 - fraction).prod(-1)
        out = (weights.unsqueeze(-1) * corner_features).sum(2)  # (N, L, F)
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
