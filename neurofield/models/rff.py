"""RFF: ReLU MLP with a random Fourier feature input encoding."""

import math
from collections.abc import Callable

import torch
from torch import Tensor, nn

from .mlp import MLP, ReLULayer

__all__ = ["RFFEncoding", "RFF"]


class RFFEncoding(nn.Module):
    """Random Fourier encoding ``[sin(2πBx), cos(2πBx)]``.

    The fixed frequency buffer ``B`` has shape ``(num_frequencies,
    in_features)`` and entries sampled from ``Normal(0, sigma**2)`` at
    construction. All sines precede all cosines; the output width is
    ``2 * num_frequencies`` and leading input dimensions are preserved.

    See Tancik et al., "Fourier Features Let Networks Learn High Frequency
    Functions in Low Dimensional Domains", NeurIPS 2020.
    """

    def __init__(
        self, in_features: int, num_frequencies: int = 256, sigma: float = 10.0
    ) -> None:
        super().__init__()

        frequencies = torch.randn(num_frequencies, in_features) * sigma
        self.register_buffer("B", frequencies)

        self.out_features = 2 * num_frequencies

    def forward(self, x: Tensor) -> Tensor:
        projection = 2 * math.pi * x @ self.B.T
        return torch.cat([torch.sin(projection), torch.cos(projection)], dim=-1)


class RFF(nn.Module):
    """ReLU MLP with random Fourier features (Tancik et al., NeurIPS 2020).

    Maps ``(..., in_features)`` to ``(..., out_features)``.

    Args:
        in_features: Number of input coordinates.
        out_features: Number of output channels.
        hidden_features: Hidden width.
        hidden_layers: Number of ReLU layers; must be at least one.
        num_frequencies: Number of random frequency vectors in :class:`RFFEncoding`.
        sigma: Standard deviation of the frequency vectors.
        output_activation: Optional callable applied to the output.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
        hidden_layers: int = 3,
        num_frequencies: int = 256,
        sigma: float = 10.0,
        output_activation: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        super().__init__()

        self.encoding = RFFEncoding(in_features, num_frequencies, sigma)
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
