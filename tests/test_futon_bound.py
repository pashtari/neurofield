"""Tests for the error bounds of FUTON with a linear decoder."""

import math

import numpy as np
import pytest
import torch
from scipy.fft import dctn

import neurofield as nf

DOUBLE = torch.float64


def signal(sizes, channels, seed=0):
    """A smooth random signal of shape ``(*sizes, channels)`` plus a little noise."""
    generator = torch.Generator().manual_seed(seed)
    axes = [torch.linspace(-1, 1, n, dtype=DOUBLE) for n in sizes]
    x = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, len(sizes))
    centers = torch.rand(6, len(sizes), dtype=DOUBLE, generator=generator) * 2 - 1
    weights = torch.randn(6, channels, dtype=DOUBLE, generator=generator)
    values = torch.exp(-torch.cdist(x, centers).square() / 0.2) @ weights
    values += 0.05 * torch.randn(values.shape, dtype=DOUBLE, generator=generator)
    return values.reshape(*sizes, channels)


def features(basis, sizes):
    """A basis's features on the grid ``linspace(-1, 1, N_c)`` of each axis."""
    return [
        basis.axis_features(torch.linspace(-1, 1, n, dtype=DOUBLE), c).double()
        for c, n in enumerate(sizes)
    ]


def futon(basis, rank, channels):
    """The paper's model: a CP combiner and a linear decoder, both without bias."""
    return nf.FUTON(
        len(basis.num_components),
        channels,
        basis,
        ("cp", {"rank": rank}),
        ("linear", {"bias": False}),
    ).double()


def grid_error(model, sizes, target):
    axes = [torch.linspace(-1, 1, n, dtype=DOUBLE) for n in sizes]
    x = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1)
    with torch.no_grad():
        return float((model(x) - target).square().mean())


@pytest.mark.parametrize(
    "sizes, channels, basis",
    [
        ((12, 10), 3, ("sinc", [6, 5])),
        ((9, 8, 7), 1, ("cosine", [4, 5, 3])),
        ((10, 9), 2, ("legendre", [5, 6])),
    ],
)
def test_upper_bound_is_attained_and_bounds_are_ordered(sizes, channels, basis):
    """A FUTON given the bound's factors has exactly the upper bound's error,
    and truncation <= lower <= upper, each nonincreasing in the rank."""
    name, components = basis
    basis = nf.FUTON(len(sizes), channels, (name, {"num_components": components}),
                     ("cp", {"rank": 1}), ("linear", {"bias": False})).basis.double()  # fmt: skip
    target = signal(sizes, channels)
    bound = nf.futon_bound(target, features(basis, sizes), ranks=[1, 2, 4, 8])
    for rank, upper in zip(bound.ranks, bound.upper):
        model = futon(basis, rank, channels)
        bound.initialize(model)
        assert grid_error(model, sizes, target) == pytest.approx(float(upper), rel=1e-9)
    assert bound.truncation <= float(bound.lower.min()) + 1e-12
    assert torch.all(bound.lower <= bound.upper + 1e-12)
    assert torch.all(bound.upper.diff() <= 1e-15)
    assert torch.all(bound.lower.diff() <= 1e-15)


def test_every_futon_is_above_the_lower_bound():
    """Random and trained FUTONs alike stay above the lower bound of their rank."""
    sizes, channels, rank = (10, 8), 3, 3
    basis = nf.SincBasis(2, num_components=[6, 5])
    target = signal(sizes, channels, seed=1)
    bound = nf.futon_bound(target, features(basis, sizes), ranks=rank)
    torch.manual_seed(0)
    for _ in range(5):
        model = futon(basis, rank, channels)
        assert grid_error(model, sizes, target) >= float(bound.lower[0])
    model = futon(basis, rank, channels)
    axes = [torch.linspace(-1, 1, n, dtype=DOUBLE) for n in sizes]
    x = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(300):
        loss = (model(x) - target).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert grid_error(model, sizes, target) >= float(bound.lower[0])


def test_matrix_case_is_exact():
    """With a full span, two axes and one channel, the best FUTON of rank R is
    the truncated SVD of the signal itself (Eckart and Young): the bounds meet
    there, whatever the basis."""
    sizes, rank = (9, 7), 3
    target = signal(sizes, 1, seed=2)
    generator = torch.Generator().manual_seed(3)
    full = [torch.randn(n, n, dtype=DOUBLE, generator=generator) for n in sizes]
    bound = nf.futon_bound(target, full, ranks=rank, sweeps=0)
    values = torch.linalg.svdvals(target[..., 0])
    best = float(values[rank:].square().sum()) / math.prod(sizes)
    assert bound.truncation == pytest.approx(0, abs=1e-12)
    assert float(bound.lower[0]) == pytest.approx(best, rel=1e-9)
    assert float(bound.upper[0]) == pytest.approx(best, rel=1e-9)


def test_bounds_depend_only_on_the_span():
    """Features Phi_c and Phi_c M_c, for invertible M_c, give the same bounds."""
    sizes, channels = (11, 9, 6), 2
    target = signal(sizes, channels, seed=4)
    generator = torch.Generator().manual_seed(5)
    phis = [
        torch.randn(n, k, dtype=DOUBLE, generator=generator)
        for n, k in zip(sizes, (6, 5, 4))
    ]
    mixed = [phi @ torch.randn(phi.shape[1], phi.shape[1], dtype=DOUBLE, generator=generator) for phi in phis]  # fmt: skip
    first = nf.futon_bound(target, phis, ranks=[2, 5], sweeps=5)
    second = nf.futon_bound(target, mixed, ranks=[2, 5], sweeps=5)
    assert first.truncation == pytest.approx(second.truncation, rel=1e-9)
    assert torch.allclose(first.lower, second.lower, rtol=1e-8)
    assert torch.allclose(first.upper, second.upper, rtol=1e-6)


def test_redundant_features_span_the_grid():
    """More functions than points: the span is the whole axis, nothing is truncated."""
    sizes, channels = (6, 5), 2
    target = signal(sizes, channels, seed=6)
    generator = torch.Generator().manual_seed(7)
    phis = [torch.randn(n, n + 3, dtype=DOUBLE, generator=generator) for n in sizes]
    bound = nf.futon_bound(target, phis, ranks=[1, 10])
    assert bound.truncation == pytest.approx(0, abs=1e-12)
    assert float(bound.upper[-1]) == pytest.approx(0, abs=1e-12)
    assert [u.shape for u in bound.factors[0][0]] == [(9, 1), (8, 1)]


def test_cosine_coefficients_are_a_dct_block():
    """The paper's cosine basis is orthonormal at pixel centres, so the
    truncation is the energy outside the leading block of the orthonormal DCT
    of the signal, which an FFT computes."""
    sizes, keep = (16, 12), (5, 4)
    target = signal(sizes, 1, seed=8)
    phis = []
    for n, k in zip(sizes, keep):
        x = (torch.arange(n, dtype=DOUBLE) + 0.5) / n  # pixel centres in [0, 1]
        scale = torch.full((k,), math.sqrt(2.0), dtype=DOUBLE)
        scale[0] = 1.0
        phis.append(
            scale * torch.cos(math.pi * torch.arange(k, dtype=DOUBLE) * x[:, None])
        )
    bound = nf.futon_bound(target, phis, ranks=1)
    coefficients = dctn(target[..., 0].numpy(), type=2, norm="ortho")
    outside = coefficients.copy()
    outside[: keep[0], : keep[1]] = 0
    assert bound.truncation == pytest.approx(
        np.square(outside).sum() / math.prod(sizes), rel=1e-9
    )


def test_initialize_rejects_other_models():
    basis = nf.SincBasis(2, num_components=[4, 4])
    bound = nf.futon_bound(signal((6, 6), 1), features(basis, (6, 6)), ranks=2)
    model = nf.FUTON(2, 1, basis, ("cp", {"rank": 2}), "linear")  # decoder bias
    with pytest.raises(ValueError, match="without bias"):
        bound.initialize(model)
