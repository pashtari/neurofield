"""F-INR: parameter count, initialization, and the CP and tensor-train combinations."""

import math

import pytest
import torch

import neurofield as nf


def parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@pytest.mark.parametrize(
    ("in_features", "out_features", "rank", "width", "layers", "mode"),
    [(2, 3, 140, 139, 3, "cp"), (3, 1, 15, 128, 3, "tt"), (3, 16, 6, 64, 2, "tt")],
)
def test_parameter_count(in_features, out_features, rank, width, layers, mode):
    """Each axis network: a sine layer from one coordinate, layers - 1 more,
    and a head of C R outputs, or C R^2 for the tensor train's core."""
    model = nf.FINR(in_features, out_features, rank, width, layers, mode)
    body = 2 * width + (layers - 1) * width * (width + 1)
    heads = [out_features * rank] * in_features
    if mode == "tt":
        heads[0] *= rank
    assert parameters(model) == in_features * body + (width + 1) * sum(heads)


def test_initialization_follows_the_authors_image_code():
    model = nf.FINR(2, 3, rank=8, hidden_features=64, hidden_layers=3)
    for network in model.axes:
        first, *hidden, head = network
        assert first.omega == 100.0 and all(layer.omega == 30.0 for layer in hidden)
        assert first.linear.weight.abs().max() <= 1.0
        bound = math.sqrt(6 / 64) / 30.0
        for linear in [layer.linear for layer in hidden] + [head]:
            assert linear.weight.abs().max() <= bound
        for layer in network:
            linear = getattr(layer, "linear", layer)
            assert torch.count_nonzero(linear.bias) == 0


@pytest.mark.parametrize(("mode", "inputs"), [("cp", 2), ("cp", 3), ("tt", 3)])
def test_combination_matches_its_formula(mode, inputs):
    torch.manual_seed(0)
    model = nf.FINR(inputs, 2, rank=4, hidden_features=16, mode=mode).double()
    x = torch.rand(10, inputs, dtype=torch.float64) * 2 - 1
    factors = [net(x[:, [i]]) for i, net in enumerate(model.axes)]
    if mode == "cp":
        expected = math.prod(f.view(10, 2, 4) for f in factors).sum(-1)
    else:
        core, left, right = (
            factors[0].view(10, 2, 4, 4),
            *(f.view(10, 2, 4) for f in factors[1:]),
        )
        expected = torch.einsum("ncpq,ncp,ncq->nc", core, left, right)
    torch.testing.assert_close(model(x), expected)


@pytest.mark.parametrize(("mode", "inputs"), [("cp", 2), ("tt", 3)])
def test_grid_evaluation_matches_per_point(mode, inputs):
    """On a grid each network runs once per distinct coordinate; the output
    and the gradients are those of running it per point."""
    torch.manual_seed(0)
    model = nf.FINR(inputs, 3, rank=8, hidden_features=32, mode=mode).double()
    axis = torch.linspace(-1, 1, 7, dtype=torch.float64)
    grid = torch.stack(torch.meshgrid(*[axis] * inputs, indexing="ij"), -1)
    grid = grid.reshape(-1, inputs)
    shared = model(grid)
    per_point = model(grid.clone().requires_grad_(True))  # bypasses the sharing
    torch.testing.assert_close(shared, per_point)
    model.grid = False  # scattered points: per point, the same output
    torch.testing.assert_close(model(grid), shared)
    params = list(model.parameters())
    for a, b in zip(
        torch.autograd.grad(shared.square().sum(), params),
        torch.autograd.grad(per_point.square().sum(), params),
    ):
        torch.testing.assert_close(a, b)


def test_output_keeps_leading_shape_and_activation():
    model = nf.FINR(2, 3, rank=4, hidden_features=8, output_activation=torch.tanh)
    out = model(torch.rand(5, 6, 2))
    assert out.shape == (5, 6, 3) and out.abs().max() < 1


@pytest.mark.parametrize(
    "kwargs",
    [{"mode": "tucker"}, {"mode": "tt"}, {"hidden_layers": 0}],
)
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        nf.FINR(2, 3, rank=4, **kwargs)
