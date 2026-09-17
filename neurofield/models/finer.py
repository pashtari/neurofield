"""FINER: coordinate MLP with variable-periodic sine activations."""

from collections.abc import Callable

import torch
from torch import Tensor, nn

from .mlp import MLP
from .siren import _siren_init_

__all__ = ["FinerLayer", "FINER"]


class FinerLayer(nn.Module):
    """Variable-periodic sine layer ``sin(omega * (abs(z) + 1) * z)``.

    Here ``z = linear(x)``. The input-dependent scale ``abs(z) + 1`` is
    detached to keep its derivative out of the gradient. Maps
    ``(..., in_features)`` to ``(..., out_features)``.
    """

    def __init__(
        self, in_features: int, out_features: int, omega: float = 30.0
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.omega = omega

        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: Tensor) -> Tensor:
        z = self.linear(x)
        scale = z.detach().abs() + 1
        return torch.sin(self.omega * scale * z)


class FINER(MLP):
    """Variable-periodic coordinate MLP (Liu et al., CVPR 2024).

    Uses :class:`FinerLayer` hidden layers and SIREN weight initialization.
    Biases keep PyTorch's defaults; the paper's enlarged first-layer bias
    range is not exposed here.

    Args:
        in_features: Number of input coordinates.
        out_features: Number of output channels.
        hidden_features: Hidden width.
        hidden_layers: Number of variable-periodic layers; must be at least one.
        omega: Base frequency multiplier in every hidden layer.
        output_activation: Optional callable applied to the output.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
        hidden_layers: int = 3,
        omega: float = 30.0,
        output_activation: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            hidden_features,
            hidden_layers,
            layer_class=FinerLayer,
            omega=omega,
            output_activation=output_activation,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights with SIREN bounds, leaving biases unchanged."""
        _siren_init_(self.layers, self.layers[0].omega)
