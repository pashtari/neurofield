"""FreSh: a SIREN's first-layer frequency chosen from the signal's spectrum."""

from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["fresh_distance", "fresh_select", "fresh_spectrum"]


def fresh_spectrum(signal: Tensor, size: int = 64) -> Tensor:
    """The spectrum FreSh compares (Kania et al., ICLR 2025), as a distribution.

    The signal, a grid of shape ``(*sides, channels)`` with two or three
    sides, is first resized to a square or a cube of its smallest side, at
    most 1024 (bilinear with antialiasing in 2D, trilinear in 3D), as the
    authors' code does. The amplitudes of its discrete Fourier transform are
    summed over the channels, and those of nonnegative frequencies are summed
    along shells of equal index sum ``d = k_1 + ... + k_n``. The shells
    ``d = 1, ..., size``, which leave out the constant term, are returned
    divided by their sum.

    Args:
        signal: Values on a grid, ``(*sides, channels)``.
        size: Number of shells, at most half the side after resizing.

    Returns:
        A ``(size,)`` float32 tensor that sums to one.
    """
    grid = signal.detach().movedim(-1, 0).float()  # (channels, *sides)
    sides = grid.shape[1:]
    if len(sides) not in (2, 3):
        raise ValueError(f"expected a 2D or 3D grid, got {len(sides)} sides")
    side = min(min(sides), 1024)
    if any(length != side for length in sides):
        planar = len(sides) == 2
        grid = F.interpolate(
            grid[None],
            size=[side] * len(sides),
            mode="bilinear" if planar else "trilinear",
            align_corners=False,
            antialias=planar,
        )[0]
    if size > side // 2:
        raise ValueError(f"{size} shells reach past half the side, {side}")
    axes = tuple(range(1, grid.ndim))
    amplitude = torch.fft.fftn(grid, dim=axes).abs().sum(0)
    corner = amplitude[(slice(0, size + 1),) * len(sides)]
    steps = torch.arange(size + 1, device=corner.device)
    shell = sum(
        steps.view([-1 if axis == i else 1 for axis in range(len(sides))])
        for i in range(len(sides))
    )
    sums = torch.zeros(len(sides) * size + 1, device=corner.device)
    sums.scatter_add_(0, shell.flatten(), corner.flatten())
    spectrum = sums[1 : size + 1]
    return spectrum / spectrum.sum()


def fresh_distance(p: Tensor, q: Tensor) -> float:
    """The Wasserstein-1 distance of two spectra on the points ``0, ..., n - 1``."""
    return float((p.cumsum(0) - q.cumsum(0))[:-1].abs().sum())


@torch.no_grad()
def fresh_select(
    targets: Sequence[Tensor],
    render: Callable[[float], Tensor],
    omegas: Sequence[float] = tuple(range(10, 201, 10)),
    size: int = 64,
    inits: int = 10,
    seed: int = 0,
) -> tuple[float, dict[float, float]]:
    """FreSh's first-layer frequency for a SIREN (Kania et al., ICLR 2025).

    For each candidate ``omega``, ``render(omega)`` builds a new, untrained
    model whose first layer has that frequency and returns its output on the
    signal's grid. It runs ``inits`` times, from the same random state for
    every candidate, and the candidate's score is the mean
    :func:`fresh_distance` between the :func:`fresh_spectrum` of the ``i``-th
    output and that of ``targets[i % len(targets)]``: one signal for a
    representation, several views or frames otherwise.

    Args:
        targets: The signal on its grid, ``(*sides, channels)``, or several.
        render: Builds an untrained model for a frequency and returns its
            output on the signal's grid.
        omegas: The candidate frequencies.
        size: The number of spectrum shells compared.
        inits: The number of initializations averaged per candidate.
        seed: The random state each candidate starts from.

    Returns:
        The candidate of least score, the smaller one on a tie, and every
        candidate's score.
    """
    goals = [fresh_spectrum(target, size) for target in targets]
    scores = {}
    for omega in omegas:
        torch.manual_seed(seed)
        distances = [
            fresh_distance(goals[i % len(goals)], fresh_spectrum(render(omega), size))
            for i in range(inits)
        ]
        scores[omega] = sum(distances) / inits
    return min(omegas, key=lambda omega: (scores[omega], omega)), scores
