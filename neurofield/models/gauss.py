"""Gauss: coordinate MLP with Gaussian activations."""

from collections.abc import Callable

import torch
from torch import Tensor, nn

from .mlp import MLP

__all__ = ["GaussLayer", "Gauss"]


class GaussLayer(nn.Module):
    """Gaussian layer ``exp(-(scale * linear(x))**2)``.

    Maps ``(..., in_features)`` to ``(..., out_features)``. Larger ``scale``
    produces narrower Gaussians.
    """

    def __init__(
        self, in_features: int, out_features: int, scale: float = 10.0
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.scale = scale

        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: Tensor) -> Tensor:
        return torch.exp(-((self.scale * self.linear(x)) ** 2))


class Gauss(MLP):
    """Gaussian coordinate MLP (Ramasinghe and Lucey, ECCV 2022).

    Uses :class:`GaussLayer` hidden layers and PyTorch's default initialization.

    Args:
        in_features: Number of input coordinates.
        out_features: Number of output channels.
        hidden_features: Hidden width.
        hidden_layers: Number of Gaussian layers; must be at least one.
        scale: Inverse Gaussian width in every hidden layer.
        output_activation: Optional callable applied to the output.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
        hidden_layers: int = 3,
        scale: float = 10.0,
        output_activation: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            hidden_features,
            hidden_layers,
            layer_class=GaussLayer,
            scale=scale,
            output_activation=output_activation,
        )
