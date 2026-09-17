"""Multi-layer perceptrons with standard or custom hidden layers."""

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, nn

__all__ = ["MLP", "ReLULayer"]


class ReLULayer(nn.Module):
    """Linear projection followed by ReLU: ``(..., in_features) -> (..., out_features)``."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: Tensor) -> Tensor:
        return torch.relu(self.linear(x))


class MLP(nn.Module):
    """MLP with standard or custom hidden layers and a linear output layer.

    Maps ``(..., in_features)`` to ``(..., out_features)``.
    By default, hidden layers are linear projections followed by ReLU.
    Subclasses apply custom initialization after ``super().__init__``.

    Args:
        in_features: Input width.
        out_features: Output width.
        hidden_features: Hidden width; ``None`` defaults to ``in_features``.
        hidden_layers: Number of hidden layers; zero gives a single linear layer
            when ``layer_class`` is omitted. Custom layers require at least one.
        activation: Callable applied after each standard hidden linear layer;
            ``None`` selects ReLU. Module instances are shared across hidden
            layers. Cannot be combined with ``layer_class``.
        layer_class: Complete hidden-layer class receiving input/output widths
            and ``**layer_kwargs``. No additional activation is applied.
        output_activation: Callable applied after the linear output layer;
            ``None`` leaves the output unchanged.
        **layer_kwargs: Constructor arguments for the hidden layers only.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int | None = None,
        hidden_layers: int = 2,
        activation: Callable[[Tensor], Tensor] | None = None,
        *,
        layer_class: type[nn.Module] | None = None,
        output_activation: Callable[[Tensor], Tensor] | None = None,
        **layer_kwargs: Any,
    ) -> None:
        super().__init__()

        min_hidden_layers = 0 if layer_class is None else 1
        if hidden_layers < min_hidden_layers:
            raise ValueError(
                f"hidden_layers must be >= {min_hidden_layers}, got {hidden_layers}"
            )
        if layer_class is None:
            layer_class = nn.Linear
            activation = torch.relu if activation is None else activation
        elif activation is not None:
            raise ValueError("Custom layer_class supplies its own activation.")

        if hidden_features is None:
            hidden_features = in_features

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.activation = activation if activation is not None else nn.Identity()

        layers = [layer_class(in_features, hidden_features, **layer_kwargs)]
        for _ in range(hidden_layers - 1):
            layers.append(layer_class(hidden_features, hidden_features, **layer_kwargs))
        layers.append(nn.Linear(hidden_features, out_features))
        self.layers = nn.ModuleList(layers)

        self.output_activation = (
            output_activation if output_activation is not None else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.output_activation(self.layers[-1](x))
