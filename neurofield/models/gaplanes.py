"""GA-Planes: geometric algebra planes grid INR."""

from collections.abc import Callable, Sequence

from torch import Tensor, nn

from .mlp import MLP
from .multivector import MultiVector

__all__ = ["GAPlanes"]


class GAPlanes(nn.Module):
    """Line, plane, and volume grid features with a ReLU MLP decoder.

    The default :class:`MultiVector` expression combines a low-rank term
    with lower-resolution grids. In 2D it concatenates the product of two
    line grids and a plane; in 3D it also includes line-plane products and a
    volume. Maps coordinates ``(..., in_features)`` in ``[-1, 1]`` to
    ``(..., out_features)``.

    Args:
        in_features: Number of input coordinates (2 or 3).
        out_features: Number of output channels.
        expr: MultiVector expression; ``None`` selects all grids.
        features: Feature dimension per grid: an int for all grids, or
            ``(line, plane, volume)`` values; short sequences repeat their last
            value. Operands of ``o`` and ``+`` must have equal channel counts.
        resolution: Resolution per axis per grid, with the same int/sequence
            rules as ``features``.
        hidden_features: Decoder hidden width.
        hidden_layers: Number of decoder hidden layers; zero gives a linear decoder.
        output_activation: Optional callable applied to the output.

    See Sivgin et al., "Geometric Algebra Planes: Convex Implicit Neural
    Volumes", arXiv 2024.
    """

    _DEFAULT_EXPR: dict[int, str] = {
        2: "[e1 o e2, e12]",
        3: "[e1 o e2 o e3, e1 o e23, e2 o e13, e3 o e12, e123]",
    }

    def __init__(
        self,
        in_features: int,
        out_features: int,
        expr: str | None = None,
        features: int | Sequence[int] = 32,
        resolution: int | Sequence[int] = (256, 64, 16),
        hidden_features: int = 64,
        hidden_layers: int = 2,
        output_activation: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        super().__init__()
        if in_features not in (2, 3):
            raise ValueError(f"in_features must be 2 or 3, got {in_features}")
        self.in_features = in_features
        self.out_features = out_features
        self.output_activation = (
            output_activation if output_activation is not None else nn.Identity()
        )

        self.mv = MultiVector(
            in_features=in_features,
            expr=self._DEFAULT_EXPR[in_features] if expr is None else expr,
            features=features,
            resolution=resolution,
        )
        self.decoder = MLP(
            self.mv.out_features, out_features, hidden_features, hidden_layers
        )

    def forward(self, x: Tensor) -> Tensor:
        features = self.mv(x.reshape(-1, self.in_features))
        out = self.output_activation(self.decoder(features))
        return out.reshape(*x.shape[:-1], self.out_features)
