"""TensoRF tensor factorizations expressed as GA-Planes grid encodings."""

from collections.abc import Callable
from typing import Literal

from torch import Tensor

from .gaplanes import GAPlanes

__all__ = ["TensoRF"]


class TensoRF(GAPlanes):
    """Tensor-factorized grid INR (Chen et al., ECCV 2022).

    A :class:`GAPlanes` model with uniform grid resolution and feature count.
    CP multiplies line grids to produce ``rank`` features. VM concatenates
    three line-plane products to produce ``3 * rank`` features; in 2D it
    falls back to CP. A ReLU MLP decodes the features.

    Maps coordinates ``(..., in_features)`` in ``[-1, 1]`` to
    ``(..., out_features)``.

    Args:
        in_features: Number of input coordinates (2 or 3).
        out_features: Number of output channels.
        rank: Feature channels per grid.
        resolution: Resolution per axis of each grid.
        mode: ``"cp"`` or ``"vm"``; ``"vm"`` becomes ``"cp"`` in 2D.
        hidden_features: Decoder hidden width.
        hidden_layers: Number of decoder hidden layers; zero gives a linear decoder.
        output_activation: Optional callable applied to the output.
    """

    _MODE_EXPR: dict[tuple[int, str], str] = {
        (2, "cp"): "e1 o e2",
        (3, "cp"): "e1 o e2 o e3",
        (3, "vm"): "[e3 o e12, e2 o e13, e1 o e23]",
    }

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        resolution: int = 128,
        mode: Literal["cp", "vm"] = "vm",
        hidden_features: int = 128,
        hidden_layers: int = 2,
        output_activation: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        if mode not in ("cp", "vm"):
            raise ValueError(f"mode must be 'cp' or 'vm', got {mode!r}")
        if in_features == 2:
            mode = "cp"  # VM reduces to CP in 2D

        super().__init__(
            in_features=in_features,
            out_features=out_features,
            expr=self._MODE_EXPR.get((in_features, mode)),
            features=rank,
            resolution=resolution,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            output_activation=output_activation,
        )

        self.resolution = resolution
        self.rank = rank
        self.mode = mode
