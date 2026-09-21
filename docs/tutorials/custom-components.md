# Custom components

The library is built from small interfaces. This page shows the contract of each one with a minimal working example.

## A basis

`FUTON` builds the basis as `basis(in_features, **kwargs)` and only requires a module that exposes `num_components`, a list with one count per axis, and returns one `(N, K_c)` feature tensor per axis from `(N, C)` coordinates. A Gaussian bump basis in a dozen lines:

```python
import torch
from torch import nn
import neurofield as nf


class GaussianBasis(nn.Module):
    """K Gaussian bumps per axis, centred on a uniform grid over [-1, 1]."""

    def __init__(self, in_features, num_components, width=0.05):
        super().__init__()
        self.in_features = in_features
        self.num_components = [num_components] * in_features
        self.width = width
        self.register_buffer("centers", torch.linspace(-1.0, 1.0, num_components))

    def forward(self, x):  # (N, C) -> C tensors of shape (N, K)
        return tuple(
            torch.exp(-(((x[:, c, None] - self.centers) / self.width) ** 2))
            for c in range(self.in_features)
        )


model = nf.FUTON(
    2, 3,
    basis=(GaussianBasis, {"num_components": 64}),
    combiner=("cp", {"rank": 32}),
    decoder="linear",
)
```

A class and a `(class, kwargs)` pair work anywhere a registry key does, as does a ready module instance. To add a key instead, register the class in `nf.models.futon.BASES`.

## A combiner or decoder

A combiner is built as `combiner(basis.num_components, **kwargs)`, takes the tuple of per-axis features, returns `(N, out_features)`, and exposes `out_features`. A decoder is any module built as `decoder(combiner.out_features, out_features, **kwargs)`, so `torch.nn.Linear`, `nf.MLP`, or a class of your own:

```python
class GatedDecoder(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.gate = nn.Linear(in_features, out_features)

    def forward(self, g):
        return self.linear(g) * torch.sigmoid(self.gate(g))


model = nf.FUTON(
    2, 3,
    basis=("sinc", {"num_components": 128}),
    combiner=("cp", {"rank": 64}),
    decoder=GatedDecoder,
)
```

## A loss

`nf.train` calls `loss_fn(batch, model)` and expects a scalar loss and a dictionary of float metrics for the log. `batch` is the dataset item with its tensors on the training device (keys with a leading underscore stay on the CPU). A loss that fits only the pixels a mask marks as known:

```python
import torch.nn.functional as F


def masked_mse(batch, model):
    output = model(batch["input"])
    mask = batch["mask"]
    loss = F.mse_loss(output[mask], batch["target"][mask])
    return loss, {"loss": loss.item()}


train_set = nf.MaskedImageCoordinateDataset("image.png", mask="mask.png", subsample=0.1)
eval_set = nf.ImageCoordinateDataset("image.png")
result = nf.train(model, train_set, eval_set, loss_fn=masked_mse)
```

The indexing works before and after subsampling because the dataset flattens `input`, `target` and `mask` with the same indices. `nf.rate_distortion_loss` and `nf.sdf_loss` are two more examples of the signature; `functools.partial(nf.rate_distortion_loss, alpha=0.01)` sets their options.

## A metric

Metrics are functions `fn(reconstruction, original) -> float` on whatever the dataset's `postprocess` returns and its `_original` holds: `uint8` `(C, H, W)` images or `{0, 1}` volumes. Rate metrics take `(encoded_bytes, original)`.

```python
def mae(pred, target):
    return (pred.float() - target.float()).abs().mean().item()


result = nf.train(model, train_set, eval_set, metrics={"psnr": nf.psnr, "mae": mae})
```

For `nf.nerf.evaluate`, metrics receive `(3, H, W)` float images in $[0, 1]$.

## A dataset

A dataset is a `torch.utils.data.Dataset` whose items carry `id`, `input`, `target` and `_original`, with `preprocess`, `postprocess` and `save` for the evaluation loop and, optionally, `to(device)` for GPU-side subsampling. The simplest route is to subclass `ImageCoordinateDataset` and override what differs; `OccupancyCoordinateDataset` and `MaskedImageCoordinateDataset` do exactly that. For a signal with its own geometry, build the coordinates with `nf.create_coordinates(size)` and follow the [conventions](../concepts/conventions.md).

## A model

Any module that maps `(..., in_features)` coordinates in $[-1, 1]$ to `(..., out_features)` is a neural field for the training loop, and, without an output activation, a density network for `nf.nerf.RadianceField`. `nf.MLP` accepts a custom `layer_class` when the novelty is in the layer:

```python
class SoftplusLayer(nn.Module):
    def __init__(self, in_features, out_features, beta=1.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.beta = beta

    def forward(self, x):
        return F.softplus(self.linear(x), beta=self.beta)


model = nf.MLP(
    2, 3, hidden_features=256, hidden_layers=3,
    layer_class=SoftplusLayer, beta=2.0, output_activation=torch.tanh,
)
```

Extra keyword arguments go to the hidden layers; the output layer stays linear. `SIREN`, `FINER`, `Gauss` and `RealWIRE` are built this way and add their initialization after `super().__init__()`.

## Registries

`nf.build_module(spec, registry, *args, **kwargs)` is the resolver behind every spec: an instance is returned as is, a key is looked up in `registry`, a class is called, and a `(key_or_class, params)` pair (a two-element list works too, so specs can come from YAML or JSON) adds keyword arguments. Use it for your own components:

```python
LOSSES = {"mse": nf.rate_distortion_loss, "sdf": nf.sdf_loss}
```

is a plain dictionary, but for modules,

```python
DECODERS = {"linear": nn.Linear, "mlp": nf.MLP, "gated": GatedDecoder}
decoder = nf.build_module(("gated", {}), DECODERS, 64, 3)
```
