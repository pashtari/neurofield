"""Radiance fields assembled from spatial, color, and direction modules.

Following Instant-NGP, the spatial network predicts density logits and
geometry features. A color network combines those features with an encoded
view direction. Holding the color network and renderer fixed lets experiments
compare spatial representations directly.
"""

from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from ..models.mlp import MLP
from ..utils import ModuleSpec, build_module

__all__ = [
    "trunc_exp",
    "SphericalHarmonicsEncoding",
    "IdentityEncoding",
    "RadianceField",
]


class _TruncExp(torch.autograd.Function):
    """Exponential with the backward input clamped to avoid large gradients."""

    @staticmethod
    def forward(ctx: Any, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tensor:
        (x,) = ctx.saved_tensors
        return grad_output * torch.exp(x.clamp(-15.0, 15.0))


def trunc_exp(x: Tensor) -> Tensor:
    """Compute ``exp(x)`` with its backward input clamped to ``[-15, 15]``.

    The forward pass is unchanged; clamping limits large density gradients
    during early training.
    """
    return _TruncExp.apply(x)


class SphericalHarmonicsEncoding(nn.Module):
    r"""Real spherical-harmonics encoding of unit directions.

    Evaluates the real SH basis for the bands ``l = 0, ..., degree - 1``,
    producing ``degree**2`` features. ``degree`` follows the tiny-cuda-nn
    convention of counting bands, so the highest spherical-harmonic degree is
    ``degree - 1``. ``degree=4`` (16 features) matches the view-direction
    encoding of Instant-NGP.

    Args:
        degree: Number of SH bands, an integer between 1 and 4.

    Shape:
        - Input: :math:`(*, 3)` unit-norm directions.
        - Output: :math:`(*, \text{degree}^2)`.
    """

    def __init__(self, degree: int = 4) -> None:
        super().__init__()
        if not isinstance(degree, int) or not 1 <= degree <= 4:
            raise ValueError(f"Expected an integer 1 <= degree <= 4, got {degree!r}")
        self.degree = degree

    @property
    def out_features(self) -> int:
        """Number of output features, ``degree**2``."""
        return self.degree**2

    def forward(self, x: Tensor) -> Tensor:
        x, y, z = x.unbind(-1)
        features = [torch.full_like(x, 0.28209479177387814)]
        if self.degree > 1:
            features += [
                -0.48860251190291987 * y,
                0.48860251190291987 * z,
                -0.48860251190291987 * x,
            ]
        if self.degree > 2:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            features += [
                1.0925484305920792 * xy,
                -1.0925484305920792 * yz,
                0.94617469575755997 * zz - 0.31539156525251999,
                -1.0925484305920792 * xz,
                0.54627421529603959 * (xx - yy),
            ]
        if self.degree > 3:
            features += [
                0.59004358992664352 * y * (-3.0 * xx + yy),
                2.8906114426405538 * xy * z,
                0.45704579946446572 * y * (1.0 - 5.0 * zz),
                0.3731763325901154 * z * (5.0 * zz - 3.0),
                0.45704579946446572 * x * (1.0 - 5.0 * zz),
                1.4453057213202769 * z * (xx - yy),
                0.59004358992664352 * x * (-xx + 3.0 * yy),
            ]
        return torch.stack(features, dim=-1)


class IdentityEncoding(nn.Module):
    """Pass unit directions through unchanged, as in FINER's NeRF network.

    Input and output have shape ``(*, 3)``; ``out_features`` is always ``3``.
    """

    out_features: int = 3

    def forward(self, x: Tensor) -> Tensor:
        return x


DIRECTION_ENCODINGS: dict[str, type[nn.Module]] = {
    "sh": SphericalHarmonicsEncoding,
    "identity": IdentityEncoding,
}

COLOR_NETS: dict[str, type[nn.Module]] = {"linear": nn.Linear, "mlp": MLP}


class RadianceField(nn.Module):
    """A radiance field with separate density, color, and direction modules.

    The density network produces one density logit followed by geometry
    features. The color network combines geometry and direction features to
    produce RGB logits. This field applies the density activation and RGB
    sigmoid, so both networks should have no output activation.

    Args:
        density_net: :data:`~neurofield.utils.ModuleSpec` for a spatial network.
            Classes and ``(class, params)`` pairs receive ``in_features=3`` and
            ``out_features=1 + geometry_features``. Instances are used as is;
            string registry keys are unsupported. Inputs lie in ``[-1, 1]^3``.
        color_net: Module spec mapping geometry and direction features to three
            RGB logits. Classes receive the input and output widths; instances
            are used as is. Registry keys: ``"linear"`` and ``"mlp"``.
            A zero-input probe checks the output width at construction.
        geometry_features: Number of features passed to the color network,
            excluding density; must be at least 1.
        direction_encoding: Module spec exposing ``out_features``. Registry
            keys: ``"sh"`` (degree-4 spherical harmonics, as in Instant-NGP)
            and ``"identity"`` (raw directions, as in FINER).
        aabb: Scene box ``(xmin, ymin, zmin, xmax, ymax, zmax)``. World points
            map affinely to ``[-1, 1]^3`` and clamp at the box boundary.
            Stored as a persistent float32 buffer in checkpoints.
        density_activation: Map density logits to nonnegative densities.
        density_bias: Offset added to density logits before activation.

    Shape:
        - Input: ``x`` and unit ``directions``, each ``(N, 3)``.
        - Output: ``(rgb, sigma)``, with RGB ``(N, 3)`` in ``[0, 1]`` and
          density ``(N,)``. Empty batches are supported.

    Example::

        field = RadianceField(
            (nf.InstantNGP, {"log2_hashmap_size": 14}),
            ("mlp", {"hidden_features": 64, "hidden_layers": 2}),
        )
        rgb, sigma = field(points, directions)
    """

    def __init__(
        self,
        density_net: ModuleSpec,
        color_net: ModuleSpec,
        *,
        geometry_features: int = 15,
        direction_encoding: ModuleSpec = "sh",
        aabb: Tensor | Sequence[float] = (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
        density_activation: Callable[[Tensor], Tensor] = trunc_exp,
        density_bias: float = 0.0,
    ) -> None:
        super().__init__()
        if geometry_features < 1:
            raise ValueError(f"geometry_features must be >= 1, got {geometry_features}")
        self.geometry_features = geometry_features
        self.density_net = build_module(
            density_net, {}, in_features=3, out_features=1 + geometry_features
        )
        self.register_buffer("aabb", torch.as_tensor(aabb, dtype=torch.float32))
        self.density_activation = density_activation
        self.density_bias = density_bias

        self.direction_encoding = build_module(direction_encoding, DIRECTION_ENCODINGS)
        color_in_features = geometry_features + self.direction_encoding.out_features
        self.color_net = build_module(
            color_net, COLOR_NETS, in_features=color_in_features, out_features=3
        )
        with torch.no_grad():
            # Supplied module instances may already use CUDA or half precision.
            parameter = next(self.color_net.parameters(), None)
            probe = torch.zeros(
                1,
                color_in_features,
                device=None if parameter is None else parameter.device,
                dtype=None if parameter is None else parameter.dtype,
            )
            out_features = int(self.color_net(probe).shape[-1])
        if out_features != 3:
            raise ValueError(
                f"color_net must map ({color_in_features},) -> (3,) RGB logits, but it "
                f"returns {out_features} features"
            )

    def normalize_coordinates(self, x: Tensor) -> Tensor:
        """Map world coordinates ``(*, 3)`` from :attr:`aabb` to ``[-1, 1]^3``.

        Points outside the box are clamped to its boundary.
        """
        box_min, box_max = self.aabb[:3], self.aabb[3:]
        return (2.0 * (x - box_min) / (box_max - box_min) - 1.0).clamp(-1.0, 1.0)

    def _density_and_features(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return densities and geometry features at world coordinates ``x``."""
        output = self.density_net(self.normalize_coordinates(x))
        sigma = self.density_activation(output[..., 0] + self.density_bias)
        return sigma, output[..., 1:]

    def query_density(self, x: Tensor) -> Tensor:
        """Return densities ``(N,)`` at world coordinates ``(N, 3)``.

        Skips the color network for occupancy updates. Empty batches are allowed.
        """
        if x.shape[0] == 0:
            return x.new_zeros((0,))
        return self._density_and_features(x)[0]

    def forward(self, x: Tensor, directions: Tensor) -> tuple[Tensor, Tensor]:
        """Return RGB and density at world points for the given unit directions."""
        if x.shape[0] == 0:
            return x.new_zeros((0, 3)), x.new_zeros((0,))
        sigma, features = self._density_and_features(x)
        direction_features = self.direction_encoding(directions)
        rgb = torch.sigmoid(
            self.color_net(torch.cat([features, direction_features], dim=-1))
        )
        return rgb, sigma
