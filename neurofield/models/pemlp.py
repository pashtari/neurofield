"""PEMLP: ReLU MLP with a NeRF-style positional encoding."""

from collections.abc import Callable

import torch
from torch import Tensor, nn

from .mlp import MLP, ReLULayer

__all__ = ["PositionalEncoding", "PEMLP"]


class PositionalEncoding(nn.Module):
    """Positional encoding ``[x, sin(2**k * x), cos(2**k * x)]``.

    Uses ``num_frequencies`` bands, with ``k`` starting at zero, without the
    π factor in NeRF Eq. (4). The input is followed by all sines, then all
    cosines; within each block, bands vary faster than coordinates.
    The output width is ``in_features * (2 * num_frequencies + 1)`` and
    leading input dimensions are preserved. Bands are stored in ``freqs``.

    See Mildenhall et al., "NeRF: Representing Scenes as Neural Radiance
    Fields for View Synthesis", ECCV 2020.
    """

    def __init__(self, in_features: int, num_frequencies: int = 10) -> None:
        super().__init__()

        freqs = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        self.register_buffer("freqs", freqs)

        self.out_features = in_features * (2 * num_frequencies + 1)

    def forward(self, x: Tensor) -> Tensor:
        projection = x.unsqueeze(-1) * self.freqs
        return torch.cat(
            [x, projection.sin().flatten(-2), projection.cos().flatten(-2)], dim=-1
        )


class PEMLP(nn.Module):
    """ReLU MLP with a NeRF-style positional encoding.

    Maps ``(..., in_features)`` to ``(..., out_features)``.

    Args:
        in_features: Number of input coordinates.
        out_features: Number of output channels.
        hidden_features: Hidden width.
        hidden_layers: Number of ReLU layers; must be at least one.
        num_frequencies: Number of bands in :class:`PositionalEncoding`.
        output_activation: Optional callable applied to the output.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
        hidden_layers: int = 3,
        num_frequencies: int = 10,
        output_activation: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        super().__init__()

        self.encoding = PositionalEncoding(in_features, num_frequencies)
        self.mlp = MLP(
            self.encoding.out_features,
            out_features,
            hidden_features,
            hidden_layers,
            layer_class=ReLULayer,
            output_activation=output_activation,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(self.encoding(x))
