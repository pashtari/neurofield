"""FreSh: the spectrum it compares, its distance, its choice, and SIREN's first omega."""

import math

import pytest
import torch

import neurofield as nf


def cosine(frequency: float, side: int = 64) -> torch.Tensor:
    """A cosine of ``frequency`` cycles along the first axis of a square grid."""
    x = torch.arange(side) / side
    return torch.cos(2 * math.pi * frequency * x)[:, None, None].expand(side, side, 1)


def test_a_cosine_is_one_shell():
    """A cosine of k cycles puts its every amplitude in shell d = k."""
    spectrum = nf.fresh_spectrum(cosine(5), size=16)
    assert spectrum.shape == (16,) and float(spectrum.sum()) == pytest.approx(1.0)
    assert float(spectrum[4]) == pytest.approx(1.0, abs=1e-5)  # float32 leakage


@pytest.mark.parametrize("shape", [(32, 32, 3), (20, 20, 20, 1)])
def test_shells_sum_nonnegative_frequencies(shape):
    """Shell d holds the amplitudes of the frequencies with index sum d."""
    torch.manual_seed(0)
    signal = torch.rand(shape)
    size = shape[0] // 2
    amplitude = torch.fft.fftn(signal.movedim(-1, 0), dim=tuple(range(1, len(shape))))
    amplitude = amplitude.abs().sum(0)
    shells = torch.zeros(size)
    for index in torch.cartesian_prod(*[torch.arange(size + 1)] * (len(shape) - 1)):
        d = int(index.sum())
        if 1 <= d <= size:
            shells[d - 1] += amplitude[tuple(index)]
    torch.testing.assert_close(nf.fresh_spectrum(signal, size), shells / shells.sum())


def test_spectrum_ignores_transposition_and_resizes_to_the_smaller_side():
    torch.manual_seed(0)
    image = torch.rand(48, 48, 3)
    torch.testing.assert_close(
        nf.fresh_spectrum(image, 16), nf.fresh_spectrum(image.transpose(0, 1), 16)
    )
    assert nf.fresh_spectrum(torch.rand(96, 64, 3), 32).shape == (32,)
    with pytest.raises(ValueError):
        nf.fresh_spectrum(torch.rand(96, 64, 3), 33)  # past half of 64


def test_distance_is_the_one_dimensional_wasserstein_distance():
    p = torch.tensor([1.0, 0.0, 0.0, 0.0])
    q = torch.tensor([0.0, 0.0, 0.0, 1.0])
    assert nf.fresh_distance(p, q) == pytest.approx(3.0)  # all mass moves 3 steps
    assert nf.fresh_distance(p, p) == 0.0


def test_select_picks_the_matching_frequency_and_the_smaller_on_a_tie():
    """A render whose cosine has omega / 10 cycles matches a 5-cycle target at 50."""
    omega, scores = nf.fresh_select(
        [cosine(5)], lambda omega: cosine(omega / 10), omegas=range(10, 101, 10),
        size=16, inits=2,
    )  # fmt: skip
    assert omega == 50 and scores[50] == pytest.approx(0.0)
    tied, _ = nf.fresh_select(
        [cosine(5)], lambda omega: cosine(5), omegas=(30, 20), size=16, inits=1
    )
    assert tied == 20


def test_select_starts_every_candidate_from_the_same_random_state():
    draws = {}

    def render(omega):
        draws.setdefault(omega, []).append(float(torch.rand(())))
        return torch.rand(32, 32, 1)

    nf.fresh_select([torch.rand(32, 32, 1)], render, omegas=(10, 20), size=8, inits=3)
    assert draws[10] == draws[20]


def test_siren_first_omega_changes_the_first_layer_alone():
    torch.manual_seed(0)
    plain = nf.SIREN(2, 3, hidden_features=16, hidden_layers=3, omega=30.0)
    torch.manual_seed(0)
    same = nf.SIREN(
        2, 3, hidden_features=16, hidden_layers=3, omega=30.0, first_omega=None
    )
    for a, b in zip(plain.parameters(), same.parameters()):
        torch.testing.assert_close(a, b)
    torch.manual_seed(0)
    model = nf.SIREN(
        2, 3, hidden_features=16, hidden_layers=3, omega=30.0, first_omega=90.0
    )
    assert model.layers[0].omega == 90.0
    assert all(layer.omega == 30.0 for layer in model.layers[1:-1])
    for a, b in zip(plain.parameters(), model.parameters()):
        torch.testing.assert_close(a, b)  # the initialization follows the later omega
