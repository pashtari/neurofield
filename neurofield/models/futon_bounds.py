r"""How well a FUTON with a linear decoder can fit a sampled signal, bounded both ways.

A FUTON with a CP combiner and a linear decoder, both without bias, is the
paper's model: its weight tensor :math:`\mathcal{W} = [\![\mathbf{U}^{(1)},
\dots, \mathbf{U}^{(C)}, \mathbf{V}]\!]` has CP rank at most ``R``. On a
signal's sampling grid, its error splits exactly in two: the error of the best
approximation in the basis's span, which no rank removes, and the CP misfit of
the signal's coefficients in an orthonormal basis of that span.
:func:`futon_bound` computes the first exactly and brackets the best second:
from below by the singular values of the coefficients' matricizations, and from
above by an explicit CP tensor, the leading terms of a nested SVD refined by
alternating least squares. Its factors, mapped back through the basis, make a
FUTON whose error is exactly the upper bound. Everything a bound rests on is
computed in float64; only the search for the factors runs in float32, since
any factors give a valid bound once their error is taken exactly. The theorem
and its proof are in the documentation, under Concepts.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import reduce
from itertools import combinations, permutations

import torch
from torch import Tensor, nn

from .futon import FUTON, CPCombiner

__all__ = ["FUTONBound", "futon_bound"]


@dataclass
class FUTONBound:
    """Bounds on the best mean squared error of a FUTON with a linear decoder.

    Errors are means over the grid's points and the signal's channels, as
    :func:`neurofield.train` reports them, and are in float64.

    Attributes:
        ranks: The CP ranks, ascending.
        truncation: The error of the best approximation in the basis's span,
            which no rank goes below; exact.
        lower: Per rank, a lower bound on the error of every FUTON of that
            rank; exact for a matrix of coefficients, that is, one axis, or two
            and one channel.
        upper: Per rank, the error of the FUTON in ``factors``, and so an
            upper bound on the best one's.
        factors: Per rank, the factor matrices ``U^(c)`` of shape ``(K_c, R)``
            and the decoder ``V`` of shape ``(D, R)`` that attain ``upper``.
    """

    ranks: list[int]
    truncation: float
    lower: Tensor
    upper: Tensor
    factors: list[tuple[list[Tensor], Tensor]]

    @torch.no_grad()
    def initialize(self, model: FUTON) -> None:
        """Give ``model`` the factors and the decoder that attain ``upper`` at its rank.

        The model must use the basis and the grid the bound was computed with;
        its error on that grid is then the bound, up to its own precision.
        """
        if not (
            isinstance(model.combiner, CPCombiner)
            and isinstance(model.decoder, nn.Linear)
            and model.combiner.linears[0].bias is None
            and model.decoder.bias is None
        ):
            raise ValueError(
                "the bound is for a FUTON with a CP combiner and a linear decoder,"
                " both without bias"
            )
        factors, decoder = self.factors[self.ranks.index(model.combiner.rank)]
        for linear, factor in zip(model.combiner.linears, factors):
            linear.weight.copy_(factor.T)
        model.decoder.weight.copy_(decoder)


def _unfold(tensor: Tensor, rows: Sequence[int]) -> Tensor:
    """The matricization with ``rows`` as row modes and the rest as columns."""
    columns = [mode for mode in range(tensor.ndim) if mode not in rows]
    shape = math.prod(tensor.shape[mode] for mode in rows)
    return tensor.permute(*rows, *columns).reshape(shape, -1)


def _tails(tensor: Tensor, ranks: Sequence[int]) -> Tensor:
    """Per rank, the most energy beyond that rank in any matricization.

    A CP tensor of rank ``R`` has matricizations of rank at most ``R``, so by
    Eckart and Young none comes closer to ``tensor`` than the energy of the
    singular values a matricization has beyond the ``R``-th.
    """
    best = torch.zeros(len(ranks), dtype=tensor.dtype)
    for size in range(1, tensor.ndim // 2 + 1):
        for rows in combinations(range(tensor.ndim), size):
            if 2 * size == tensor.ndim and 0 not in rows:  # each split once
                continue
            matrix = _unfold(tensor, rows)
            if matrix.shape[0] > matrix.shape[1]:
                matrix = matrix.T
            energies = torch.linalg.eigvalsh(matrix @ matrix.T).flip(0).clamp_min(0)
            beyond = energies.sum() - torch.cat(
                [energies.new_zeros(1), energies.cumsum(0)]
            )
            tails = beyond[[min(rank, len(energies)) for rank in ranks]].cpu()
            best = torch.maximum(best, tails)
    return best


def _expansion(tensor: Tensor, order: Sequence[int]) -> tuple[Tensor, list[Tensor]]:
    """The orthonormal rank-1 expansion of a batch of tensors, by nested SVDs.

    ``tensor`` is ``(B, I_1, ..., I_M)``. The SVD of each tensor's
    matricization with ``order[0]`` as rows writes it as a sum of orthogonal
    terms, each a unit vector times a unit tensor of the other modes, which
    are expanded in turn, all at once as a batch. The result is a sum of
    mutually orthogonal rank-1 tensors of unit norm, each weighted by a
    product of singular values. Returns the weights ``(B, L)`` and, per mode,
    the unit vectors ``(B, L, I_m)`` of the terms.
    """
    batch, *shape = tensor.shape
    first = order[0]
    rest = [mode for mode in range(len(shape)) if mode != first]
    matrix = tensor.permute(0, first + 1, *(mode + 1 for mode in rest))
    u, s, vh = torch.linalg.svd(
        matrix.reshape(batch, shape[first], -1), full_matrices=False
    )
    terms = s.shape[1]
    if len(rest) == 1:
        vectors = [u.transpose(1, 2), vh] if first == 0 else [vh, u.transpose(1, 2)]
        return s, vectors
    inner = vh.reshape(batch * terms, *(shape[mode] for mode in rest))
    weights, parts = _expansion(inner, [rest.index(mode) for mode in order[1:]])
    length = weights.shape[1]
    vectors: list[Tensor] = [Tensor()] * len(shape)
    for mode, part in zip(rest, parts):
        vectors[mode] = part.reshape(batch, terms * length, -1)
    vectors[first] = (
        u.transpose(1, 2).reshape(batch * terms, 1, -1).expand(-1, length, -1)
    ).reshape(batch, terms * length, -1)
    return (s.reshape(batch * terms, 1) * weights).reshape(batch, -1), vectors


def _residual(tensor: Tensor, factors: Sequence[Tensor]) -> float:
    """``||tensor - [[factors]]||_F^2``, with the CP tensor formed in full."""
    letters = "abcdefgh"[: tensor.ndim]
    spec = f"{','.join(letter + 'r' for letter in letters)}->{letters}"
    return float((tensor - torch.einsum(spec, *factors)).square().sum())


def _sweep(tensor: Tensor, factors: list[Tensor]) -> list[Tensor]:
    """One sweep of alternating least squares: each mode's factor solved exactly."""
    letters = "abcdefgh"[: tensor.ndim]
    for mode in range(tensor.ndim):
        others = [k for k in range(tensor.ndim) if k != mode]
        gram = reduce(torch.mul, (factors[k].T @ factors[k] for k in others))
        ridge = 1e-12 * gram.diagonal().mean().clamp_min(1e-300)
        spec = (
            f"{letters},{','.join(letters[k] + 'r' for k in others)}->{letters[mode]}r"
        )
        right = torch.einsum(spec, tensor, *(factors[k] for k in others))
        eye = torch.eye(len(gram), dtype=gram.dtype, device=gram.device)
        factors[mode] = torch.linalg.solve(gram + ridge * eye, right.T).T
    # Spread each component's scale evenly over the modes; the tensor is unchanged.
    norms = torch.stack([factor.norm(dim=0) for factor in factors]).clamp_min(1e-150)
    scale = norms.log().mean(0).exp()
    return [factor / norm * scale for factor, norm in zip(factors, norms)]


def _als(
    tensor: Tensor, factors: list[Tensor], sweeps: int, tol: float, floor: float
) -> list[Tensor]:
    """Alternating least squares from ``factors``, until the bound settles.

    After each sweep, the step it took is extended by ``k^(1/3)`` at sweep
    ``k`` and kept if that lowers the residual: Bro's line search, which
    shortens the slow stretches of ALS. Neither move raises the residual, and
    the sweeps stop once one lowers it by less than ``tol`` of the bound,
    ``floor`` plus the residual, that it is part of.
    """
    error = _residual(tensor, factors)
    for k in range(1, sweeps + 1):
        update = _sweep(tensor, list(factors))
        trial = [new + k ** (1 / 3) * (new - old) for new, old in zip(update, factors)]
        candidates = [
            (_residual(tensor, update), update),
            (_residual(tensor, trial), trial),
        ]
        best_error, best = min(candidates, key=lambda candidate: candidate[0])
        if best_error > error:  # rounding at convergence
            break
        settled = error - best_error < tol * (floor + best_error)
        error, factors = best_error, best
        if settled:
            break
    return factors


@torch.no_grad()
def futon_bound(
    signal: Tensor,
    features: Sequence[Tensor],
    ranks: int | Sequence[int],
    sweeps: int = 500,
    tol: float = 3e-5,
) -> FUTONBound:
    """Bound the best grid error of a FUTON with a linear decoder on a signal.

    Every FUTON with a CP combiner and a linear decoder, both without bias,
    with these per-axis features and rank ``R``, has a mean squared error on
    the grid of at least ``lower[R]``, and the one given by ``factors[R]`` has
    exactly ``upper[R]``; no FUTON goes below ``truncation``. The bounds
    depend on the basis only through each axis's span, so any basis of the
    same span gives the same ones.

    Args:
        signal: Samples of shape ``(N_1, ..., N_C, D)`` on a grid.
        features: Per axis ``c``, the basis at that axis's grid coordinates,
            of shape ``(N_c, K_c)``: any functions, orthonormal or not. For a
            FUTON's basis on a neurofield dataset's grid, it is
            ``basis.axis_features(torch.linspace(-1, 1, N_c), c)``.
        ranks: The CP ranks ``R`` to bound.
        sweeps: Most alternating least-squares sweeps that refine each upper
            bound, in float32, from its SVD form, the ``R`` heaviest terms of
            the nested SVDs; ``0`` keeps the SVD form.
        tol: The sweeps stop once one lowers the bound by less than this
            fraction of it.

    Returns:
        The bounds and the factors that attain the upper ones, on the CPU.

    Example::

        data = nf.ImageCoordinateDataset("kodim19.png")
        H, W, _ = data.target.shape
        basis = nf.SincBasis(2, num_components=[H // 2, W // 2])
        features = [
            basis.axis_features(torch.linspace(-1, 1, n), c)
            for c, n in enumerate((H, W))
        ]
        bound = nf.futon_bound(data.target, features, ranks=[64, 256])
        psnr = 10 * torch.log10(4 / bound.upper)  # guaranteed, in dB
    """
    ranks = sorted({ranks} if isinstance(ranks, int) else set(ranks))
    signal = signal.double()
    *sizes, channels = signal.shape
    if len(features) != len(sizes) or any(
        len(phi) != size for phi, size in zip(features, sizes)
    ):
        raise ValueError(
            f"expected one feature matrix of N_c rows per axis of {tuple(sizes)}"
        )

    # Phi_c / sqrt(N_c) = Q_c B_c with Q_c an orthonormal basis of its span;
    # directions the grid cannot tell from zero are left out. The small
    # decompositions here and below run on the CPU, where they are fastest.
    bases, inverses = [], []
    for phi in features:
        phi = phi.double().cpu() / math.sqrt(len(phi))
        q, s, vh = torch.linalg.svd(phi, full_matrices=False)
        keep = s > s[0] * max(phi.shape) * torch.finfo(s.dtype).eps
        bases.append(q[:, keep])
        inverses.append(vh[keep].T / s[keep])  # B_c^+, (K_c, K_c')

    # The signal's coefficients in the orthonormal basis of the span.
    coefficients = signal
    for axis, q in enumerate(bases):
        coefficients = torch.tensordot(coefficients, q.to(signal), dims=([axis], [0]))
        coefficients = coefficients.movedim(-1, axis)
    coefficients = coefficients / math.sqrt(math.prod(sizes))
    energy = float(coefficients.square().sum())
    truncation = max(float(signal.square().mean()) * channels - energy, 0.0)

    small = coefficients.cpu()
    lower = _tails(small, ranks)

    # The upper bounds' SVD form: per rank, the R heaviest terms of the nested
    # SVD expansion along the best mode order. Modes of size one are peeled
    # first, at no cost, and the last two of an order share one SVD.
    modes = range(coefficients.ndim)
    single = [mode for mode in modes if coefficients.shape[mode] == 1]
    rest = [mode for mode in modes if coefficients.shape[mode] > 1]
    orders = [
        single + list(order)
        for order in permutations(rest)
        if len(order) < 2 or order[-2] < order[-1]
    ]
    expansions = []
    for order in orders:
        weights, vectors = _expansion(small.unsqueeze(0), order)
        weights, index = weights[0].sort(descending=True)
        vectors = [part[0, index] for part in vectors]
        kept = torch.cat([weights.new_zeros(1), weights.square().cumsum(0)])
        expansions.append((weights, vectors, kept))

    # The search runs in float32 on the signal's device; each bound is the
    # exact float64 error of the factors it finds.
    working = coefficients.float()
    upper, factors = [], []
    for rank in ranks:
        errors = [
            energy - float(kept[min(rank, len(kept) - 1)]) for _, _, kept in expansions
        ]
        weights, vectors, _ = expansions[
            min(range(len(errors)), key=errors.__getitem__)
        ]
        terms = min(rank, len(weights))
        initial = [part[:terms].T.to(working) for part in vectors]
        initial[0] = initial[0] * weights[:terms].to(working)
        chosen = _als(working, initial, sweeps, tol, truncation)
        chosen = [factor.double() for factor in chosen]
        residual = _residual(coefficients, chosen)
        # A rank's FUTON is one of every larger rank, with zero components added.
        if upper and upper[-1] < residual:
            residual, chosen = upper[-1], factors[-1]
        padded = [
            torch.cat([f, f.new_zeros(len(f), rank - f.shape[1])], 1) for f in chosen
        ]
        upper.append(residual)
        factors.append(padded)

    # Back to the basis: U^(c) = B_c^+ U~^(c), with the decoder unchanged.
    mapped = [
        (
            [inverse @ factor.cpu() for inverse, factor in zip(inverses, chosen[:-1])],
            chosen[-1].cpu(),
        )
        for chosen in factors
    ]
    to_mse = 1 / channels
    return FUTONBound(
        ranks=ranks,
        truncation=truncation * to_mse,
        lower=(truncation + lower) * to_mse,
        upper=(truncation + torch.tensor(upper, dtype=torch.float64)) * to_mse,
        factors=mapped,
    )
