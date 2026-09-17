"""WIRE: coordinate MLPs with Gabor wavelet activations (real and complex)."""

import math
from collections.abc import Callable

import torch
from torch import Tensor, nn

from .mlp import MLP

__all__ = ["RealGaborLayer", "ComplexGaborLayer", "RealWIRE", "WIRE"]


class RealGaborLayer(nn.Module):
    """Real Gabor layer ``cos(omega * z1) * exp(-(scale * z2)**2)``.

    A linear projection produces ``z1`` and ``z2`` with ``out_features``
    channels each. ``omega`` controls frequency and ``scale`` controls the
    inverse Gaussian width. Leading input dimensions are preserved.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        omega: float = 20.0,
        scale: float = 10.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.omega = omega
        self.scale = scale

        self.linear = nn.Linear(in_features, 2 * out_features)

    def forward(self, x: Tensor) -> Tensor:
        projection = self.linear(x)
        frequency_projection, scale_projection = projection.chunk(2, dim=-1)
        return torch.cos(self.omega * frequency_projection) * torch.exp(
            -((self.scale * scale_projection) ** 2)
        )


class ComplexGaborLayer(nn.Module):
    """Complex Gabor layer ``exp(1j * omega * z - abs(scale * z)**2)``.

    Here ``z = linear(x)``. The projection uses real weights when
    ``is_first=True`` and complex weights otherwise. Maps
    ``(..., in_features)`` to complex ``(..., out_features)`` values;
    ``omega`` controls frequency and ``scale`` the inverse Gaussian width.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        omega: float = 20.0,
        scale: float = 10.0,
        is_first: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.omega = omega
        self.scale = scale

        dtype = torch.float if is_first else torch.cfloat
        self.linear = nn.Linear(in_features, out_features, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        projection = self.linear(x)
        return torch.exp(
            1j * self.omega * projection - (self.scale * projection).abs().square()
        )


class RealWIRE(MLP):
    """Real-valued variant of :class:`WIRE` with :class:`RealGaborLayer`.

    Uses PyTorch's default initialization and a linear output layer. The
    hidden width is reduced as in WIRE; each real Gabor layer projects to
    twice that width to form its two wavelet terms.

    Args:
        in_features: Number of input coordinates.
        out_features: Number of output channels.
        hidden_features: Nominal width, divided by ``sqrt(2)`` and rounded down.
        hidden_layers: Number of Gabor layers; must be at least one.
        omega: Frequency multiplier.
        scale: Inverse Gaussian width.
        output_activation: Optional callable applied to the output.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
        hidden_layers: int = 3,
        omega: float = 20.0,
        scale: float = 10.0,
        output_activation: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        hidden_features = int(hidden_features / math.sqrt(2))

        super().__init__(
            in_features,
            out_features,
            hidden_features,
            hidden_layers,
            layer_class=RealGaborLayer,
            omega=omega,
            scale=scale,
            output_activation=output_activation,
        )


class WIRE(nn.Module):
    """Wavelet implicit neural representation (Saragadam et al., CVPR 2023).

    Uses a real first projection, complex Gabor hidden layers, and a complex
    linear output layer, all with PyTorch's default initialization. Returns
    the real part of the output, followed by ``output_activation``.
    Maps real ``(..., in_features)`` to real ``(..., out_features)`` values.

    Args:
        in_features: Number of input coordinates.
        out_features: Number of output channels.
        hidden_features: Nominal width, divided by ``sqrt(2)`` and rounded down
            to keep the parameter count comparable to real-valued networks.
        hidden_layers: Number of Gabor layers; must be at least one.
        omega: Frequency multiplier.
        scale: Inverse Gaussian width.
        output_activation: Optional callable applied to the output.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
        hidden_layers: int = 3,
        omega: float = 20.0,
        scale: float = 10.0,
        output_activation: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        super().__init__()
        if hidden_layers < 1:
            raise ValueError(f"hidden_layers must be >= 1, got {hidden_layers}")

        # Account for the two real components of each complex parameter.
        hidden_features = int(hidden_features / math.sqrt(2))

        self.layers = nn.ModuleList()
        self.layers.append(
            ComplexGaborLayer(
                in_features, hidden_features, omega=omega, scale=scale, is_first=True
            )
        )
        for _ in range(hidden_layers - 1):
            self.layers.append(
                ComplexGaborLayer(
                    hidden_features, hidden_features, omega=omega, scale=scale
                )
            )
        self.layers.append(nn.Linear(hidden_features, out_features, dtype=torch.cfloat))

        self.output_activation = (
            output_activation if output_activation is not None else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.output_activation(x.real)
