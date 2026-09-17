"""SIREN: coordinate MLP with sine activations."""

import math
from collections.abc import Callable

import torch
from torch import Tensor, nn

from .mlp import MLP

__all__ = ["SineLayer", "SIREN"]


class SineLayer(nn.Module):
    """Sine layer ``sin(omega * linear(x))`` used by :class:`SIREN`.

    Maps ``(..., in_features)`` to ``(..., out_features)``; ``omega`` controls
    frequency.
    """

    def __init__(
        self, in_features: int, out_features: int, omega: float = 30.0
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.omega = omega

        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(self.omega * self.linear(x))


class SIREN(MLP):
    """Sinusoidal representation network (Sitzmann et al., NeurIPS 2020).

    An :class:`MLP` with sine hidden layers, a linear output layer,
    and SIREN weight initialization. Biases keep PyTorch's defaults.

    Args:
        in_features: Number of input coordinates.
        out_features: Number of output channels.
        hidden_features: Hidden width.
        hidden_layers: Number of sine layers; must be at least one.
        omega: Frequency multiplier in every sine layer.
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
            layer_class=SineLayer,
            omega=omega,
            output_activation=output_activation,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights with SIREN bounds, leaving biases unchanged."""
        _siren_init_(self.layers, self.layers[0].omega)


@torch.no_grad()
def _siren_init_(layers: nn.ModuleList, omega: float) -> None:
    """Initialize SIREN-style weights in place (Sitzmann et al., Sec. 3.2).

    The first layer uses ``U(-1/n, 1/n)`` and later layers, including the
    linear output, ``U(-sqrt(6/n)/omega, sqrt(6/n)/omega)``, where ``n`` is
    the layer's input width. Biases are left unchanged.
    """
    for i, layer in enumerate(layers):
        if i == 0:
            bound = 1 / layer.in_features
        else:
            bound = math.sqrt(6 / layer.in_features) / omega
        getattr(layer, "linear", layer).weight.uniform_(-bound, bound)
