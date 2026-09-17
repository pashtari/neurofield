"""Behavior and checkpoint layouts of plain and custom-layer MLPs."""

import math
from functools import partial

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import neurofield as nf


@pytest.mark.parametrize("activation", [None, nn.PReLU(init=0.25)])
def test_hidden_activation_is_shared(activation):
    model = nf.MLP(1, 1, 1, 2, activation)
    with torch.no_grad():
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                layer.weight.fill_(1)
                layer.bias.zero_()

    output = model(torch.tensor([[-2.0], [2.0]]))
    expected = [[0.0], [2.0]] if activation is None else [[-0.125], [2.0]]
    torch.testing.assert_close(output, torch.tensor(expected))
    if activation is not None:
        assert model.get_submodule("activation") is activation
        assert [
            name
            for name, parameter in model.named_parameters(remove_duplicate=False)
            if parameter is activation.weight
        ] == ["activation.weight"]
        output.sum().backward()
        torch.testing.assert_close(activation.weight.grad, torch.tensor([-1.0]))


@pytest.mark.parametrize("activation", [None, torch.tanh])
def test_omitted_hidden_width_defaults_to_input_width(activation):
    model = nf.MLP(2, 3, activation=activation)
    reference = nf.MLP(2, 3, hidden_features=2, activation=activation)
    reference.load_state_dict(model.state_dict())
    x = torch.tensor([[-2.0, 1.0], [0.5, -1.0]])

    assert model.hidden_features == 2
    torch.testing.assert_close(model(x), reference(x), rtol=0, atol=0)


def test_hidden_activation_leaves_linear_output_signed():
    model = nf.MLP(1, 1, 1, 1, activation=torch.relu)
    model.load_state_dict(
        {
            "layers.0.weight": torch.ones(1, 1),
            "layers.0.bias": torch.zeros(1),
            "layers.1.weight": -torch.ones(1, 1),
            "layers.1.bias": torch.zeros(1),
        }
    )

    torch.testing.assert_close(model(torch.tensor([[2.0]])), torch.tensor([[-2.0]]))


def test_hidden_layer_kwargs_do_not_affect_output_layer():
    model = nf.MLP(2, 1, 3, 2, bias=False)

    assert all(layer.bias is None for layer in model.layers[:-1])
    assert model.layers[-1].bias is not None
    assert model(torch.zeros(4, 2)).shape == (4, 1)


@pytest.mark.parametrize(
    ("activation", "module_activation"),
    [
        (torch.tanh, nn.Tanh()),
        (F.relu, nn.ReLU()),
        (partial(F.leaky_relu, negative_slope=0.2), nn.LeakyReLU(0.2)),
        (lambda x: torch.sigmoid(x), nn.Sigmoid()),
    ],
    ids=["torch", "functional", "partial", "lambda"],
)
def test_callable_hidden_activations_match_modules(activation, module_activation):
    model = nf.MLP(2, 3, 4, 2, activation=activation).double()
    reference = nf.MLP(2, 3, 4, 2, activation=module_activation).double()
    assert model.state_dict().keys() == reference.state_dict().keys()
    reference.load_state_dict(model.state_dict())
    x = torch.tensor(
        [[-2.0, 1.0], [0.5, -1.0]], dtype=torch.float64, requires_grad=True
    )
    reference_x = x.detach().clone().requires_grad_()

    output = model(x)
    reference_output = reference(reference_x)
    torch.testing.assert_close(output, reference_output, rtol=0, atol=0)
    output.square().sum().backward()
    reference_output.square().sum().backward()
    torch.testing.assert_close(x.grad, reference_x.grad, rtol=0, atol=0)
    for parameter, reference_parameter in zip(
        model.parameters(), reference.parameters(), strict=True
    ):
        torch.testing.assert_close(
            parameter.grad, reference_parameter.grad, rtol=0, atol=0
        )


@pytest.mark.parametrize("output_activation", [torch.sigmoid, nn.PReLU(init=0.25)])
def test_output_activation_accepts_callable_or_module(output_activation):
    model = nf.MLP(
        1,
        1,
        hidden_layers=1,
        activation=nn.Identity(),
        output_activation=output_activation,
    )
    with torch.no_grad():
        for layer in model.layers:
            layer.weight.fill_(1)
            layer.bias.zero_()
    x = torch.tensor([[[-2.0], [2.0]]])

    output = model(x)
    torch.testing.assert_close(output, output_activation(x))
    if isinstance(output_activation, nn.Module):
        assert model.get_submodule("output_activation") is output_activation
        assert "output_activation.weight" in model.state_dict()
        output.sum().backward()
        torch.testing.assert_close(output_activation.weight.grad, torch.tensor([-2.0]))


def test_custom_layers_own_their_activation_and_receive_kwargs():
    model = nf.MLP(1, 1, 1, 2, layer_class=nf.SineLayer, omega=2.0)
    with torch.no_grad():
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                layer.weight.fill_(1)
                layer.bias.zero_()
    x = torch.tensor([[[-0.5], [0.5]]])

    torch.testing.assert_close(model(x), torch.sin(2 * torch.sin(2 * x)))


@pytest.mark.parametrize("activation", [nn.ReLU(), torch.relu])
def test_custom_layers_reject_an_additional_hidden_activation(activation):
    with pytest.raises(ValueError, match="activation"):
        nf.MLP(2, 1, activation=activation, layer_class=nf.SineLayer)


@pytest.mark.parametrize("layer_class", [None, nf.SineLayer])
@pytest.mark.parametrize("hidden_layers", [0, -1])
def test_invalid_hidden_layer_counts(hidden_layers, layer_class):
    with pytest.raises(ValueError, match="hidden_layers must be >= 1"):
        nf.MLP(2, 1, hidden_layers=hidden_layers, layer_class=layer_class)


@pytest.mark.parametrize(
    ("layer_class", "hidden_prefix", "output_prefix"),
    [(None, "layers.0", "layers.1"), (nf.ReLULayer, "layers.0.linear", "layers.1")],
)
def test_plain_and_custom_checkpoint_layouts(layer_class, hidden_prefix, output_prefix):
    checkpoint = {
        f"{hidden_prefix}.weight": torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        f"{hidden_prefix}.bias": torch.zeros(3),
        f"{output_prefix}.weight": torch.tensor([[1.0, 2.0, 3.0]]),
        f"{output_prefix}.bias": torch.tensor([0.5]),
    }
    model = nf.MLP(2, 1, 3, 1, layer_class=layer_class)
    model.load_state_dict(checkpoint)

    x = torch.tensor([[1.0, 2.0], [-2.0, 1.0]])
    torch.testing.assert_close(model(x), torch.tensor([[14.5], [2.5]]))


@pytest.mark.parametrize("model_class", [nf.SIREN, nf.FINER])
@pytest.mark.parametrize("reset", [False, True])
def test_sine_models_initialize_and_reset_weights_without_changing_biases(
    model_class, reset
):
    model = model_class(2, 1, 8, hidden_layers=2, omega=30.0)
    linear_layers = [layer for layer in model.modules() if isinstance(layer, nn.Linear)]
    biases = [layer.bias.detach().clone() for layer in linear_layers]
    if reset:
        with torch.no_grad():
            for layer in linear_layers:
                layer.weight.fill_(99)
        model.reset_parameters()

    for index, (layer, bias) in enumerate(zip(linear_layers, biases)):
        bound = (
            1 / layer.in_features
            if index == 0
            else math.sqrt(6 / layer.in_features) / 30
        )
        assert layer.weight.abs().max().item() <= bound
        torch.testing.assert_close(layer.bias, bias, rtol=0, atol=0)
