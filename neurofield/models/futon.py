"""FUTON: coordinate bases, tensor-network combiners, and the FUTON model.

FUTON evaluates ``decoder(combiner(basis(x)))`` in three stages:

1. A **basis** maps coordinates ``(N, C)`` in ``[-1, 1]`` to one feature tensor
   ``(N, K_c)`` per axis, where ``K_c = num_components[c]``.
2. A **combiner** fuses the per-axis features into one ``(N, out_features)``
   tensor.
3. A **decoder**, ``nn.Linear`` or :class:`MLP`, maps the fused features to the
   output.

All bases share three options:

- ``num_components``: an int shared by all axes, or one count per axis.
- ``normalize``: L2-normalize each feature vector (not each basis function).
- ``grid_size``: tabulate each axis on ``linspace(-1, 1, size)`` at
  construction. Coordinates on the grid read the table and others are evaluated
  directly. Table lookups carry no coordinate gradients, so leave this ``None``
  when gradients with respect to ``x`` are needed.

The local bases :class:`TriangleBasis` and :class:`LanczosBasis` default to
sparse mode: each axis returns an ``(N, K_c)`` :class:`RCSMatrix` that stores
only the few nonzero taps per point, and the combiners contract it directly.
"""

import math
from collections.abc import Callable, Sequence
from functools import reduce
from operator import mul

import opt_einsum as oe
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..utils import ModuleSpec, build_module
from .mlp import MLP
from .rcs_matrix import RCSMatrix, local_taps, rcs_product

__all__ = [
    "CosineBasis",
    "SincBasis",
    "LegendreBasis",
    "ChebyshevBasis",
    "TriangleBasis",
    "LanczosBasis",
    "HadamardCombiner",
    "CPCombiner",
    "TRCombiner",
    "TuckerCombiner",
    "FUTON",
]


def _broadcast(value: int | Sequence[int], length: int, name: str) -> list[int]:
    """Expand a shared int to ``length`` values, or check a per-axis sequence."""
    values = [value] * length if isinstance(value, int) else list(value)
    if len(values) != length:
        raise ValueError(f"{name} must have length {length}, got {len(values)}")
    return values


def _grid_position(x: Tensor, size: int) -> Tensor:
    """Map coordinates in ``[-1, 1]`` to continuous indices in ``[0, size - 1]``."""
    return (x + 1) / 2 * (size - 1)


# Bases --------------------------------------------------------------------------------


class _Basis(nn.Module):
    """Base class for coordinate bases.

    Subclasses implement :meth:`_evaluate` for one axis and call
    :meth:`_build_cache` once the buffers it needs are registered.
    """

    # Distance, in grid steps, within which a coordinate reads the cached table.
    _grid_atol = 1e-4

    def __init__(
        self,
        in_features: int,
        num_components: int | Sequence[int],
        normalize: bool = True,
        min_components: int = 1,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_components = _broadcast(num_components, in_features, "num_components")
        if min(self.num_components) < min_components:
            raise ValueError(
                f"num_components must be >= {min_components}, "
                f"got {self.num_components}"
            )
        self.normalize = normalize
        self.grid_size: list[int] | None = None

    def _evaluate(self, x: Tensor, axis: int) -> Tensor:
        """Evaluate one axis at coordinates ``(*,)``, returning ``(*, K_axis)``."""
        raise NotImplementedError

    def _normalize(self, features: Tensor) -> Tensor:
        return F.normalize(features, dim=-1) if self.normalize else features

    def _build_cache(self, grid_size: int | Sequence[int] | None) -> None:
        """Tabulate every axis on a regular grid; ``None`` disables the cache."""
        if grid_size is None:
            return
        self.grid_size = _broadcast(grid_size, self.in_features, "grid_size")
        for axis, size in enumerate(self.grid_size):
            table = self._evaluate(torch.linspace(-1.0, 1.0, size), axis)
            self.register_buffer(f"_cache_{axis}", table, persistent=False)

    def _lookup(self, x: Tensor, axis: int, size: int) -> Tensor:
        """Read on-grid coordinates from the cache and evaluate the rest."""
        if size == 1:
            return self._evaluate(x, axis)

        position = _grid_position(x, size)
        index = position.round()
        on_grid = (
            ((position - index).abs() < self._grid_atol)
            & (position >= -self._grid_atol)
            & (position <= size - 1 + self._grid_atol)
        )
        features = self.get_buffer(f"_cache_{axis}")[index.clamp(0, size - 1).long()]
        if not bool(on_grid.all()):
            off_grid = ~on_grid
            features[off_grid] = self._evaluate(x[off_grid], axis)
        return features

    def axis_features(self, x: Tensor, axis: int) -> Tensor:
        """The dense features ``(*, K_axis)`` of one axis at coordinates ``(*,)``.

        Coordinates on a cached grid read its table; the others are evaluated.
        """
        if self.grid_size is None:
            return self._evaluate(x, axis)
        return self._lookup(x, axis, self.grid_size[axis])

    def _features(self, x: Tensor) -> tuple[Tensor, ...]:
        return tuple(
            self.axis_features(x[..., axis], axis) for axis in range(self.in_features)
        )

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected last dimension {self.in_features}, got {x.shape[-1]}"
            )
        return self._features(x)


class CosineBasis(_Basis):
    r"""Cosine basis :math:`\cos(k \pi u)` with :math:`u = (x + 1) / 2`.

    Axis ``c`` uses frequencies :math:`k = 0, \dots, K_c - 1`. This is the
    FUTON paper's basis without its :math:`\sqrt{2}` factor; see
    :class:`FUTON`.

    Args:
        in_features: Number of coordinate axes ``C``.
        num_components: Number of frequencies ``K_c``, shared or per axis.
        normalize: L2-normalize each feature vector.
        grid_size: Per-axis cache size; see the module notes.

    Shape:
        - Input: :math:`(*, C)`.
        - Output: ``C`` tensors of shape :math:`(*, K_c)`.
    """

    frequencies: Tensor

    def __init__(
        self,
        in_features: int,
        num_components: int | Sequence[int],
        normalize: bool = True,
        grid_size: int | Sequence[int] | None = None,
    ) -> None:
        super().__init__(in_features, num_components, normalize)
        frequencies = math.pi * torch.arange(max(self.num_components))
        self.register_buffer("frequencies", frequencies, persistent=False)
        self._build_cache(grid_size)

    def _evaluate(self, x: Tensor, axis: int) -> Tensor:
        u = ((x + 1) / 2).unsqueeze(-1)
        frequencies = self.frequencies[: self.num_components[axis]]
        return self._normalize(torch.cos(frequencies * u))


class SincBasis(_Basis):
    r"""Cardinal sine basis centered on a uniform grid over ``[-1, 1]``.

    Component ``k`` is :math:`\operatorname{sinc}((x - \mu_k) / w)`, with
    centers :math:`\mu_k = -1 + k w` and spacing :math:`w = 2 / (K_c - 1)`.
    Each component equals one at its own center and zero at all other
    centers, so ``normalize=False`` gives cardinal (Whittaker) interpolation.

    Args:
        in_features: Number of coordinate axes ``C``.
        num_components: Number of centers ``K_c >= 2``, shared or per axis.
        normalize: L2-normalize each feature vector.
        grid_size: Per-axis cache size; see the module notes.

    Shape:
        - Input: :math:`(*, C)`.
        - Output: ``C`` tensors of shape :math:`(*, K_c)`.
    """

    def __init__(
        self,
        in_features: int,
        num_components: int | Sequence[int],
        normalize: bool = True,
        grid_size: int | Sequence[int] | None = None,
    ) -> None:
        super().__init__(in_features, num_components, normalize, min_components=2)
        self.width = [2.0 / (count - 1) for count in self.num_components]
        for axis, count in enumerate(self.num_components):
            centers = torch.linspace(-1.0, 1.0, count)
            self.register_buffer(f"centers_{axis}", centers, persistent=False)
        self._build_cache(grid_size)

    def _evaluate(self, x: Tensor, axis: int) -> Tensor:
        centers = self.get_buffer(f"centers_{axis}")
        return self._normalize(
            torch.sinc((x.unsqueeze(-1) - centers) / self.width[axis])
        )


def _chebyshev_to_legendre(count: int) -> Tensor:
    """The ``(count, count)`` matrix ``M`` with ``P_n = sum_k M[n, k] T_k``.

    Bonnet's recurrence, :math:`k P_k = (2k - 1) x P_{k-1} - (k - 1) P_{k-2}`,
    run on Chebyshev coefficients with :math:`x T_j = (T_{j+1} + T_{j-1}) / 2`.
    The entries are nonnegative and every row sums to one, so the map is as
    well conditioned as the polynomials themselves.
    """
    matrix = torch.zeros(count, count, dtype=torch.float64)
    matrix[0, 0] = 1.0
    if count > 1:
        matrix[1, 1] = 1.0
    for n in range(2, count):
        previous = matrix[n - 1]
        times_x = torch.zeros_like(previous)
        times_x[1:] += previous[:-1] / 2
        times_x[:-1] += previous[1:] / 2
        times_x[1] += previous[0] / 2  # x T_0 = T_1
        matrix[n] = ((2 * n - 1) * times_x - (n - 1) * matrix[n - 2]) / n
    return matrix.to(torch.get_default_dtype())


class ChebyshevBasis(_Basis):
    r"""Chebyshev polynomials of the first kind on ``[-1, 1]``.

    Degree ``k`` is :math:`T_k(x) = \cos(k \arccos x)`, so every degree
    comes from one cosine over a ``(*, K_c)`` grid. The unscaled polynomials
    are orthogonal under the weight :math:`(1 - x^2)^{-1/2}`.

    Args:
        in_features: Number of coordinate axes ``C``.
        num_components: Number of polynomials ``K_c``, shared or per axis.
        normalize: L2-normalize each feature vector.
        grid_size: Per-axis cache size; see the module notes.

    Shape:
        - Input: :math:`(*, C)`.
        - Output: ``C`` tensors of shape :math:`(*, K_c)`.
    """

    def __init__(
        self,
        in_features: int,
        num_components: int | Sequence[int],
        normalize: bool = True,
        grid_size: int | Sequence[int] | None = None,
    ) -> None:
        super().__init__(in_features, num_components, normalize)
        self._build_cache(grid_size)

    def _polynomials(self, x: Tensor, axis: int) -> Tensor:
        count = self.num_components[axis]
        degrees = torch.arange(count, dtype=x.dtype, device=x.device)
        return torch.cos(degrees * torch.arccos(x.unsqueeze(-1)))

    def _evaluate(self, x: Tensor, axis: int) -> Tensor:
        return self._normalize(self._polynomials(x, axis))


class LegendreBasis(ChebyshevBasis):
    r"""Legendre polynomials on ``[-1, 1]``.

    Each is a fixed combination of Chebyshev polynomials, with nonnegative
    coefficients that sum to one, so the basis is the Chebyshev one followed
    by a matrix product; see :func:`_chebyshev_to_legendre`. The polynomials
    are the standard, unscaled orthogonal ones.

    Args:
        in_features: Number of coordinate axes ``C``.
        num_components: Number of polynomials ``K_c``, shared or per axis.
        normalize: L2-normalize each feature vector.
        grid_size: Per-axis cache size; see the module notes.

    Shape:
        - Input: :math:`(*, C)`.
        - Output: ``C`` tensors of shape :math:`(*, K_c)`.
    """

    def __init__(
        self,
        in_features: int,
        num_components: int | Sequence[int],
        normalize: bool = True,
        grid_size: int | Sequence[int] | None = None,
    ) -> None:
        _Basis.__init__(self, in_features, num_components, normalize)
        matrix = _chebyshev_to_legendre(max(self.num_components))
        self.register_buffer("_from_chebyshev", matrix, persistent=False)
        self._build_cache(grid_size)

    def _polynomials(self, x: Tensor, axis: int) -> Tensor:
        count = self.num_components[axis]
        matrix = self._from_chebyshev[:count, :count]
        return super()._polynomials(x, axis) @ matrix.T


class _LocalBasis(_Basis):
    """Compact kernels centered on ``linspace(-1, 1, K_c)``.

    Subclasses implement :meth:`_kernel` on offsets ``t`` measured in grid
    steps. The kernel must vanish for ``|t| >= radius``, so each coordinate
    touches at most ``2 * radius`` consecutive components; sparse mode
    evaluates only those taps, in one fused kernel on CUDA when the subclass
    names its kernel in ``_tap_kernel``.
    """

    _tap_kernel: int | None = None  # the kernel's id for :func:`local_taps`

    def __init__(
        self,
        in_features: int,
        num_components: int | Sequence[int],
        radius: int,
        normalize: bool = True,
        grid_size: int | Sequence[int] | None = None,
        sparse: bool = True,
    ) -> None:
        if radius < 1:
            raise ValueError(f"radius must be >= 1, got {radius}")
        super().__init__(
            in_features, num_components, normalize, min_components=2 * radius
        )
        self.radius = int(radius)
        self.sparse = bool(sparse)
        if not self.sparse:
            self._build_cache(grid_size)
        # Sparse mode evaluates every axis at once, with per-axis sizes.
        sizes = torch.tensor(self.num_components, dtype=torch.float32)
        self.register_buffer("_sizes", sizes, persistent=False)
        self.register_buffer("_taps", torch.arange(2.0 * radius), persistent=False)

    def _kernel(self, t: Tensor) -> Tensor:
        raise NotImplementedError

    def _evaluate(self, x: Tensor, axis: int) -> Tensor:
        size = self.num_components[axis]
        centers = torch.arange(size, device=x.device, dtype=x.dtype)
        t = _grid_position(x, size).unsqueeze(-1) - centers
        return self._normalize(self._kernel(t))

    def _features(self, x: Tensor) -> tuple[Tensor, ...]:
        if not self.sparse:
            return super()._features(x)
        # Evaluate the 2 * radius taps around each point, on every axis at once.
        x = x.reshape(-1, self.in_features)
        if x.is_cuda and local_taps is not None and self._tap_kernel is not None:
            values, start = local_taps(
                x, self._sizes, self.radius, self._tap_kernel, self.normalize
            )
        else:
            position = _grid_position(x, self._sizes)
            # Clamping keeps each segment inside the basis; taps shifted by
            # the clamp lie outside the kernel support and evaluate to zero.
            start = (position.floor() - (self.radius - 1)).clamp_(min=0)
            start = start.minimum(self._sizes - 2 * self.radius)
            t = (position - start).unsqueeze(-1) - self._taps
            values, start = self._normalize(self._kernel(t)), start.long()
        # The clamp keeps the segments in bounds, so skip the synchronizing check.
        return tuple(
            RCSMatrix(values[:, axis], start[:, axis], size, check_invariants=False)
            for axis, size in enumerate(self.num_components)
        )


class TriangleBasis(_LocalBasis):
    r"""Triangle (hat) basis for piecewise-linear interpolation on ``[-1, 1]``.

    Component ``k`` is :math:`\max(0, 1 - |x - \mu_k| / w)` with centers spaced
    by :math:`w = 2 / (K_c - 1)`, so at most two components are nonzero. With
    ``normalize=False``, a linear map of these features equals sampling a 1D
    feature grid with ``align_corners=True``; a CP combiner and MLP decoder
    then give the TensoRF-CP parametrization.

    Args:
        in_features: Number of coordinate axes ``C``.
        num_components: Number of centers ``K_c >= 2``, shared or per axis.
        normalize: L2-normalize each feature vector.
        grid_size: Per-axis cache size in dense mode; ignored when sparse.
        sparse: Return :class:`RCSMatrix` features with two taps per point.

    Shape:
        - Input: :math:`(*, C)` in dense mode, :math:`(N, C)` in sparse mode.
        - Output: ``C`` features of shape :math:`(*, K_c)`, or
          :math:`(N, K_c)` RCS matrices in sparse mode.

    References:
        Chen et al., "TensoRF: Tensorial Radiance Fields", ECCV 2022.
    """

    def __init__(
        self,
        in_features: int,
        num_components: int | Sequence[int],
        normalize: bool = True,
        grid_size: int | Sequence[int] | None = None,
        sparse: bool = True,
    ) -> None:
        super().__init__(in_features, num_components, 1, normalize, grid_size, sparse)

    _tap_kernel = 0

    def _kernel(self, t: Tensor) -> Tensor:
        return (1.0 - t.abs()).clamp(min=0.0)


class LanczosBasis(_LocalBasis):
    r"""Windowed-sinc (Lanczos) basis centered on a uniform grid over ``[-1, 1]``.

    With grid offset :math:`t = (x - \mu_k) / w` and radius :math:`a`, the
    kernel is :math:`\operatorname{sinc}(t) \operatorname{sinc}(t / a)` for
    :math:`|t| < a` and zero elsewhere. Each component equals one at its own
    center and zero at all other centers; at most ``2 * radius`` are nonzero.

    Args:
        in_features: Number of coordinate axes ``C``.
        num_components: Number of centers ``K_c >= 2 * radius``, shared or per
            axis.
        radius: Kernel radius in grid steps: ``2`` gives Lanczos-2 (four taps),
            ``3`` (the default) gives Lanczos-3 (six taps).
        normalize: L2-normalize each feature vector.
        grid_size: Per-axis cache size in dense mode; ignored when sparse.
        sparse: Return :class:`RCSMatrix` features with ``2 * radius`` taps per
            point.

    Shape:
        - Input: :math:`(*, C)` in dense mode, :math:`(N, C)` in sparse mode.
        - Output: ``C`` features of shape :math:`(*, K_c)`, or
          :math:`(N, K_c)` RCS matrices in sparse mode.
    """

    def __init__(
        self,
        in_features: int,
        num_components: int | Sequence[int],
        radius: int = 3,
        normalize: bool = True,
        grid_size: int | Sequence[int] | None = None,
        sparse: bool = True,
    ) -> None:
        super().__init__(
            in_features, num_components, radius, normalize, grid_size, sparse
        )

    _tap_kernel = 1

    def _kernel(self, t: Tensor) -> Tensor:
        return torch.where(
            t.abs() < self.radius,
            torch.sinc(t) * torch.sinc(t / self.radius),
            t.new_zeros(()),
        )


# Combiners ----------------------------------------------------------------------------


def _project(features: Tensor, linear: nn.Linear) -> Tensor:
    """Apply ``linear`` to dense or RCS features without densifying the latter."""
    if isinstance(features, RCSMatrix):
        out = features @ linear.weight.T
        return out if linear.bias is None else out + linear.bias
    return linear(features)


class _Combiner(nn.Module):
    """Base class for combiners; ``in_features[c]`` is the width ``K_c`` of axis ``c``."""

    out_features: int

    def __init__(self, in_features: Sequence[int]) -> None:
        super().__init__()
        self.in_features = list(in_features)


class HadamardCombiner(_Combiner):
    """Elementwise product of per-axis features that share one width ``K``.

    RCS features are densified before multiplication.

    Args:
        in_features: Feature width of each axis; all entries must be equal.

    Shape:
        - Input: ``C`` tensors of shape :math:`(*, K)`.
        - Output: :math:`(*, K)`.
    """

    def __init__(self, in_features: Sequence[int]) -> None:
        super().__init__(in_features)
        if len(set(self.in_features)) != 1:
            raise ValueError(
                f"Expected all in_features entries to be equal, got {self.in_features}"
            )
        self.out_features = self.in_features[0]

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        return reduce(
            mul,
            (f.to_dense() if isinstance(f, RCSMatrix) else f for f in features),
        )


class CPCombiner(_Combiner):
    r"""Canonical polyadic (CP) combiner.

    Projects each axis to ``rank`` channels and multiplies the projections
    elementwise. ``linears[c].weight`` stores the transposed CP factor
    :math:`\mathbf{U}^{(c)\top}` of shape ``(rank, K_c)`` (FUTON paper,
    Eqs. 10–11). RCS features are projected without densifying, and without
    bias, one fused :func:`rcs_product` computes the whole combination.

    Args:
        in_features: Feature width ``K_c`` of each axis.
        rank: CP rank, which is also the output width.
        bias: Add a bias to each projection; the paper's model has none.

    Shape:
        - Input: ``C`` tensors of shape :math:`(N, K_c)`.
        - Output: :math:`(N, \text{rank})`.

    References:
        Kolda and Bader, "Tensor Decompositions and Applications", SIAM Review 2009.
    """

    def __init__(
        self, in_features: Sequence[int], rank: int, bias: bool = False
    ) -> None:
        super().__init__(in_features)
        self.rank = rank
        self.out_features = rank
        self.linears = nn.ModuleList(
            nn.Linear(width, rank, bias=bias) for width in self.in_features
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize factors with Kaiming-uniform weights and zero biases."""
        for linear in self.linears:
            nn.init.kaiming_uniform_(linear.weight)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        if self.linears[0].bias is None and all(
            isinstance(feature, RCSMatrix) for feature in features
        ):
            return rcs_product(features, [linear.weight.T for linear in self.linears])
        projections = (
            _project(feature, linear) for feature, linear in zip(features, self.linears)
        )
        return reduce(mul, projections)


class TRCombiner(_Combiner):
    r"""Tensor-ring (TR) combiner.

    Axis ``c`` contracts its features with a core of shape
    ``(rank, K_c, rank)`` into a ``rank x rank`` matrix. The ordered product of
    these matrices is flattened to ``rank**2`` features, leaving the ring
    closure (e.g. a trace) to the decoder. This extends the paper's CP model.

    Args:
        in_features: Feature width ``K_c`` of each axis.
        rank: Tensor-ring rank; the output width is ``rank**2``.

    Shape:
        - Input: ``C`` tensors of shape :math:`(N, K_c)`.
        - Output: :math:`(N, \text{rank}^2)`.

    References:
        Zhao et al., "Tensor Ring Decomposition", arXiv 2016.
    """

    def __init__(self, in_features: Sequence[int], rank: int) -> None:
        super().__init__(in_features)
        self.rank = rank
        self.out_features = rank**2
        self.cores = nn.ParameterList(
            nn.Parameter(torch.empty(rank, width, rank)) for width in self.in_features
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the cores with Kaiming-uniform weights."""
        for core in self.cores:
            nn.init.kaiming_uniform_(core)

    @staticmethod
    def _transfer(features: Tensor, core: Tensor) -> Tensor:
        """Contract ``(N, K)`` features with an ``(R, K, R)`` core into ``(N, R, R)``."""
        if isinstance(features, RCSMatrix):
            # Fold the rank axes so the contraction is one sparse matmul.
            rank, width, _ = core.shape
            flat = features @ core.permute(1, 0, 2).reshape(width, rank * rank)
            return flat.unflatten(-1, (rank, rank))
        return torch.einsum("...k,rks->...rs", features, core)

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        matrices = map(self._transfer, features, self.cores)
        return reduce(torch.matmul, matrices).flatten(start_dim=-2)


class TuckerCombiner(_Combiner):
    r"""Tucker combiner.

    Projects axis ``c`` to ``rank[c]`` channels, as the CP combiner does, and
    contracts the projections with a core of shape
    ``(rank[0], ..., rank[C - 1], out_features)`` instead of multiplying them
    elementwise: with a linear decoder the weight tensor is in Tucker format,
    the core with one factor matrix per axis and the decoder as the last.
    CP is the special case of a superdiagonal core. This extends the paper's
    CP model. RCS features are projected without densifying, and the core
    absorbs one projection at a time, largest rank first.

    Args:
        in_features: Feature width ``K_c`` of each axis.
        rank: Multilinear rank, one per axis or one for all.
        out_features: Width of the output, the core's last mode; ``None``
            takes the largest rank.

    Shape:
        - Input: ``C`` tensors of shape :math:`(N, K_c)`.
        - Output: :math:`(N, \text{out\_features})`.

    References:
        Tucker, "Some mathematical notes on three-mode factor analysis",
        Psychometrika 1966.
    """

    def __init__(
        self,
        in_features: Sequence[int],
        rank: int | Sequence[int],
        out_features: int | None = None,
    ) -> None:
        super().__init__(in_features)
        modes = len(self.in_features)
        self.rank = [rank] * modes if isinstance(rank, int) else list(rank)
        if len(self.rank) != modes:
            raise ValueError(f"Expected {modes} ranks, one per axis, got {self.rank}")
        self.out_features = max(self.rank) if out_features is None else out_features
        self.linears = nn.ModuleList(
            nn.Linear(width, r, bias=False) for width, r in zip(in_features, self.rank)
        )
        self.core = nn.Parameter(torch.empty(*self.rank, self.out_features))
        # The core absorbs one projection at a time, largest rank first, so
        # the intermediates stay smallest and lead with the points (the
        # ellipsis); opt_einsum's own optimum starts with an outer product of
        # the projections, slower and larger. Each result goes to the end of
        # the operands, hence the path; given the path, the points' count of 1
        # is a placeholder.
        self.order = sorted(range(modes), key=self.rank.__getitem__, reverse=True)
        self.contract = oe.contract_expression(
            tuple(self.core.shape),
            [*range(modes + 1)],
            *(arg for c in self.order for arg in ((1, self.rank[c]), [..., c])),
            [..., modes],
            optimize=[(0, 1)] + [(0, n) for n in range(modes - 1, 0, -1)],
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the factors and the core with Kaiming-uniform weights."""
        for linear in self.linears:
            nn.init.kaiming_uniform_(linear.weight)
        nn.init.kaiming_uniform_(self.core.view(-1, self.out_features))

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        projections = [
            _project(feature, linear) for feature, linear in zip(features, self.linears)
        ]
        return self.contract(self.core, *(projections[c] for c in self.order))


# FUTON --------------------------------------------------------------------------------


class FUTON(nn.Module):
    r"""Fourier Tensor Network for implicit neural representations.

    Computes ``output_activation(decoder(combiner(basis(x))))``. Each stage is
    a module instance, a registry key, a module class, or a
    ``(key_or_class, kwargs)`` pair. Registry keys:

    - Bases: ``"cosine"``, ``"sinc"``, ``"legendre"``, ``"chebyshev"``,
      ``"triangle"``, ``"lanczos"``.
    - Combiners: ``"hadamard"``, ``"cp"``, ``"tr"``.
    - Decoders: ``"linear"``, ``"mlp"``.

    Args:
        in_features: Number of coordinate axes ``C``.
        out_features: Number of output channels.
        basis: Basis spec, built as ``basis(in_features, **kwargs)``. Custom
            bases must expose ``num_components``, one count per axis.
        combiner: Combiner spec, built as
            ``combiner(basis.num_components, **kwargs)``. Custom combiners must
            expose ``out_features``.
        decoder: Decoder spec, built as
            ``decoder(combiner.out_features, out_features, **kwargs)``.
        output_activation: Callable applied to the output; ``None`` uses the
            identity.

    Shape:
        - Input: :math:`(*, C)` coordinates in ``[-1, 1]``.
        - Output: :math:`(*, \text{out\_features})`.

    Relation to the paper:
        The cosine basis, CP combiner, and linear decoder implement Eq. (9):
        ``combiner.linears[c].weight`` stores :math:`\mathbf{U}^{(c)\top}` and
        ``decoder.weight`` stores :math:`\mathbf{V}`; the decoder bias is an
        addition. Coordinates are mapped internally from ``[-1, 1]`` to
        ``[0, 1]``. The basis omits the paper's :math:`\sqrt{2}` factor and
        L2-normalizes features by default; ``normalize=False`` recovers Eq. (1)
        up to per-frequency constants absorbed by the factors. Experiments use
        ``torch.tanh`` at the output. The other bases, the Hadamard and
        tensor-ring combiners, and the MLP decoder are extensions.

    Example::

        model = FUTON(
            in_features=2,
            out_features=3,
            basis=("cosine", {"num_components": 256}),
            combiner=("cp", {"rank": 256}),
            decoder="linear",
            output_activation=torch.tanh,
        )
        rgb = model(torch.rand(64, 64, 2) * 2 - 1)  # (64, 64, 3)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        basis: ModuleSpec,
        combiner: ModuleSpec,
        decoder: ModuleSpec,
        output_activation: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.basis = build_module(basis, BASES, in_features)
        self.combiner = build_module(combiner, COMBINERS, self.basis.num_components)
        self.decoder = build_module(
            decoder, DECODERS, self.combiner.out_features, out_features
        )
        self.output_activation = (
            output_activation if output_activation is not None else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        # Sparse bases expect a flat batch of points.
        features = self.basis(x.reshape(-1, self.in_features))
        out = self.output_activation(self.decoder(self.combiner(features)))
        return out.reshape(*x.shape[:-1], self.out_features)


# Registries ---------------------------------------------------------------------------

BASES: dict[str, type[nn.Module]] = {
    "cosine": CosineBasis,
    "sinc": SincBasis,
    "legendre": LegendreBasis,
    "chebyshev": ChebyshevBasis,
    "triangle": TriangleBasis,
    "lanczos": LanczosBasis,
}

COMBINERS: dict[str, type[nn.Module]] = {
    "hadamard": HadamardCombiner,
    "cp": CPCombiner,
    "tr": TRCombiner,
    "tucker": TuckerCombiner,
}

DECODERS: dict[str, type[nn.Module]] = {
    "linear": nn.Linear,
    "mlp": MLP,
}
