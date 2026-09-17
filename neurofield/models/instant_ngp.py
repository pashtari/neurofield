"""Pure-PyTorch multiresolution hash encoding and Instant-NGP INR.

Follows the paper's grid convention: resolutions count cells, with one extra
vertex per axis. The CUDA reference uses a different alignment and table
padding, so its checkpoints and outputs are not interchangeable.
"""

import math
from collections.abc import Callable
from itertools import product

import torch
from torch import Tensor, nn

from .mlp import MLP

__all__ = ["HashEncoding", "InstantNGP"]

# Spatial-hash primes from the Instant-NGP reference implementation. The first
# coordinate is deliberately unhashed (prime 1) for cache coherence.
_PRIMES = (1, 2654435761, 805459861)


class HashEncoding(nn.Module):
    """Multiresolution hash encoding (Müller et al., SIGGRAPH 2022).

    Each level interpolates features from neighboring grid vertices. Grids
    that fit within the table limit use dense indexing; finer grids use a
    spatial hash. Features are concatenated across levels, with output width
    ``num_levels * features_per_level``. Inputs have shape
    ``(..., in_features)`` and values in ``[-1, 1]``.

    Args:
        in_features: Number of input coordinates (1, 2, or 3).
        num_levels: Number of resolution levels.
        features_per_level: Feature channels per table entry.
        log2_hashmap_size: Base-2 logarithm of the maximum entries per level.
        base_resolution: Cells per axis at the coarsest level.
        max_resolution: Target cells per axis at the finest level; ignored
            for a single level. Geometric growth is floored, so floating-point
            rounding can put the finest level one cell below this target.

    The learnable ``embeddings`` table concatenates all levels and starts
    uniformly in ``[-1e-4, 1e-4]``. ``resolutions`` stores cells per axis.
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
        self.hashmap_size = 2**log2_hashmap_size

        if num_levels > 1:
            growth = math.exp(
                (math.log(max_resolution) - math.log(base_resolution))
                / (num_levels - 1)
            )
        else:
            growth = 1.0
        resolutions = [
            math.floor(base_resolution * growth**level) for level in range(num_levels)
        ]
        # Dense indexing when the level's full vertex grid fits in the table.
        table_sizes = [
            min((resolution + 1) ** in_features, self.hashmap_size)
            for resolution in resolutions
        ]
        offsets = [0]
        for size in table_sizes:
            offsets.append(offsets[-1] + size)

        # Python scalars avoid device synchronization inside the level loop.
        self.resolutions = resolutions
        self.table_sizes = table_sizes
        self.offsets = offsets
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
            torch.empty(offsets[-1], features_per_level).uniform_(-1e-4, 1e-4)
        )

    @property
    def out_features(self) -> int:
        """Number of output features, ``num_levels * features_per_level``."""
        return self.num_levels * self.features_per_level

    def forward(self, x: Tensor) -> Tensor:
        batch_shape = x.shape[:-1]
        x = x.reshape(-1, self.in_features)
        unit_x = (x + 1.0) / 2.0

        features: list[Tensor] = []
        for resolution, table_size, offset in zip(
            self.resolutions, self.table_sizes, self.offsets
        ):
            position = unit_x * resolution
            base = position.floor().long().clamp(0, resolution - 1)  # (N, C)
            fraction = position - base.float()  # (N, C) in [0, 1]

            vertices = base.unsqueeze(1) + self.corners  # (N, 2^C, C)
            if table_size == (resolution + 1) ** self.in_features:
                # Dense: row-major linear index over the vertex grid.
                index = vertices[..., 0]
                for dim in range(1, self.in_features):
                    index = index * (resolution + 1) + vertices[..., dim]
            else:
                hashed_axes = (vertices * self.primes).unbind(-1)
                hashed = hashed_axes[0]
                for value in hashed_axes[1:]:
                    hashed = torch.bitwise_xor(hashed, value)
                index = hashed % table_size

            corner_features = self.embeddings[offset + index]  # (N, 2^C, F)
            # Multilinear interpolation multiplies the weights along each axis.
            weights = torch.where(
                self.corners.bool(), fraction.unsqueeze(1), 1.0 - fraction.unsqueeze(1)
            ).prod(-1)
            features.append((weights.unsqueeze(-1) * corner_features).sum(1))

        out = torch.cat(features, dim=-1)
        return out.reshape(*batch_shape, self.out_features)


class InstantNGP(nn.Module):
    """Multiresolution hash encoding followed by a ReLU MLP.

    Maps coordinates ``(..., in_features)`` in ``[-1, 1]`` to
    ``(..., out_features)``. See :class:`HashEncoding` for grid conventions
    and encoding parameters.

    Args:
        in_features: Number of input coordinates (1, 2, or 3).
        out_features: Number of output channels.
        num_levels: Number of encoding levels.
        features_per_level: Feature channels per table entry.
        log2_hashmap_size: Base-2 logarithm of the maximum entries per level.
        base_resolution: Cells per axis at the coarsest level.
        max_resolution: Target cells per axis at the finest level.
        hidden_features: Decoder hidden width.
        hidden_layers: Number of decoder hidden layers; zero gives a linear decoder.
        output_activation: Optional callable applied to the output.
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
            self.encoding.out_features, out_features, hidden_features, hidden_layers
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.output_activation(self.decoder(self.encoding(x)))
