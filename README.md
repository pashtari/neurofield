# NeuroField

A PyTorch library for neural fields (implicit neural representations). NeuroField provides modular implementations of neural field architectures, including FUTON (Fourier Tensor Network), along with training, evaluation, quantization, image compression and novel view synthesis tooling.

## Features

- **Architectures** (`nf.*`) — FUTON; coordinate networks: SIREN, FINER, Gauss, WIRE / RealWIRE, RFF, PEMLP, MFN (plus the generic `MLP`); grid and tensor models: InstantNGP, TensoRF, GAPlanes, GridINR; Deep Image Prior: DIPUNet, DIPSkip
- **Datasets** — coordinate datasets for images (`ImageCoordinateDataset`, `MaskedImageCoordinateDataset`), 3D occupancy volumes (`OccupancyCoordinateDataset`), MRI k-space (`MRICoordinateDataset`), and a noise-input dataset for Deep Image Prior (`DIPImageDataset`)
- **Unified training & evaluation** — common `nf.train` / `nf.evaluate` API across models, with `nf.chunked_inference` for memory-bounded inference
- **Quantization-aware training** — uniform quantization with a configurable integer range (`quant_max`)
- **Image compression** (`nf.compression`) — FUTON codec (`encode_futon` / `decode_futon`), PIL codecs as baselines (`encode_pil` / `decode_pil`), `evaluate_compression` (PSNR, bits per pixel, timings), and arithmetic coding
- **Novel view synthesis** (`nf.nerf`) — radiance fields built from any point-wise neurofield model, occupancy-grid volume rendering (nerfacc when available, pure PyTorch otherwise), Blender scenes, and PSNR / SSIM / LPIPS evaluation

## Installation

**Prerequisites:** Python ≥ 3.11, PyTorch ≥ 2.10

```bash
git clone https://github.com/pashtari/neurofield.git
cd neurofield
pip install -e .
```

For 3D tasks (marching cubes, mesh I/O, rendering):

```bash
pip install -e ".[3d]"
```

For novel view synthesis (nerfacc, LPIPS, video export):

```bash
pip install -e ".[nerf]"
```

## Quick Start

```python
import torch
import neurofield as nf

model = nf.FUTON(
    in_features=2,
    out_features=3,
    basis=("cosine", {"num_components": 256}),
    combiner=("cp", {"rank": 256}),
    decoder="linear",
    output_activation=torch.tanh,
)

x = torch.rand(10, 2) * 2 - 1  # 10 coordinates in [-1, 1]^2
y = model(x)                   # (10, 3)
```

FUTON is assembled from three specs, each a registry key, a `(key, kwargs)` pair, an `nn.Module` subclass, or an `nn.Module` instance:

- `basis`: `"cosine"`, `"sinc"`, `"legendre"`, `"chebyshev"`, `"triangle"`, `"lanczos"`
- `combiner`: `"hadamard"`, `"cp"`, `"tr"`
- `decoder`: `"linear"`, `"mlp"`

### Training on an image

```python
import torch
import neurofield as nf

train_dataset = nf.ImageCoordinateDataset("path/to/image.png", subsample=0.1)
eval_dataset = nf.ImageCoordinateDataset("path/to/image.png")

model = nf.FUTON(
    in_features=2,
    out_features=3,
    basis=("cosine", {"num_components": 256}),
    combiner=("cp", {"rank": 256}),
    decoder="linear",
    output_activation=torch.tanh,
)

results = nf.train(
    model,
    train_dataset,
    eval_dataset,
    num_epochs=1000,
    lr=1e-2,
    metrics={"psnr": nf.psnr, "ssim": nf.ssim},
    log_interval=100,
    eval_interval=100,
    log_dir="logs/futon",
)  # {"config": ..., "model_dict": ..., "history": [...]}

# Reconstruct the full image, 65536 coordinates per forward pass
output = nf.chunked_inference(model, eval_dataset.input, chunk_size=65536)  # (H, W, 3)
reconstruction = eval_dataset.postprocess(output.cpu())  # uint8 (3, H, W)
print(nf.psnr(reconstruction, eval_dataset.original))
```

### Image compression

```python
from torchvision.io import ImageReadMode, read_image
import neurofield as nf

image = read_image("path/to/image.png", mode=ImageReadMode.RGB)  # uint8 (3, H, W)

futon = nf.compression.evaluate_compression(
    image,
    nf.compression.encode_futon,
    nf.compression.decode_futon,
    num_epochs=500,
    basis=("cosine", {"num_components": 128}),
    combiner=("cp", {"rank": 64}),
    decoder="linear",
)
jpeg = nf.compression.evaluate_compression(
    image, nf.compression.encode_pil, nf.compression.decode_pil, format="JPEG", quality=50
)
print(futon)  # {"PSNR (dB)": ..., "bit rate (bpp)": ..., "encoding time (ms)": ..., "decoding time (ms)": ...}
```

### Novel view synthesis

```python
import neurofield as nf

root = "data/nerf/blender/lego"
train_dataset = nf.nerf.BlenderDataset(root, split="train", downsample=4, skip=4)
test_dataset = nf.nerf.BlenderDataset(root, split="test", downsample=4)

field = nf.nerf.RadianceField(
    nf.InstantNGP,  # density net
    ("mlp", {"hidden_features": 64, "hidden_layers": 2}),  # color net
    aabb=train_dataset.aabb,
)
renderer = nf.nerf.create_renderer(
    aabb=train_dataset.aabb, near=train_dataset.near, far=train_dataset.far
)

results = nf.nerf.train(field, renderer, train_dataset, num_steps=20000)
scores = nf.nerf.evaluate(field, renderer, test_dataset)  # PSNR and SSIM by default
print(scores["mean"])
```

## Benchmarks

`scripts/train_<task>.py` fits every model in `configs/<task>.yaml` to every signal of a task:

```bash
python scripts/train_image.py       # 24 Kodak images    -> logs/image/<image>/<model>/
python scripts/train_occupancy.py   # 5 Stanford shapes  -> logs/occupancy/<shape>/<model>/
python scripts/train_nerf.py        # Blender scenes     -> logs/nerf/<scene>/<model>/
```

`--data`, `--models`, `--device`, `--log-dir`, and `--config` select the signals, models, device, output directory, and config, e.g. `python scripts/train_image.py --data data/Kodak/kodim01.png --models SIREN FINER --device cuda:1`. By default, the scripts use every signal found under `data/`. Finished runs are skipped, so an interrupted sweep can simply be restarted; a failed run writes `error.txt` and the sweep continues.

The YAML configs hold the shared data and training settings and, per model, its `neurofield` class, constructor arguments, and training overrides (e.g. the learning rate). Image configs may use size expressions in the image height `H` and width `W`, such as `max(H, W) // 2`. Learning rates are drawn from {3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4}.

The runs of `configs/<name>.yaml` go to `logs/<name>/`. Runs are keyed by model name, so variants of a model need their own names or their own `--log-dir`. `configs/ablation-futon/` holds the four FUTON ablations on the occupancy task: the basis, components against rank, the tensor network, and the decoder. `--set` overrides a single value, and `--overwrite` reruns finished runs instead of skipping them:

```bash
python scripts/train_occupancy.py --config configs/ablation-futon/basis.yaml  # -> logs/ablation-futon/basis/
python scripts/train_image.py --set models.SIREN.train.lr=0.001 --log-dir logs/siren-lr
```

Each run directory holds `log.txt`, `log.json`, `checkpoint.pt`, the final reconstruction (NeRF: selected test views), and `results.json` with the model setup, parameter count, training time, final metrics (NeRF: all 200 test views), and training history.

`scripts/report_paper.py` turns those runs into `results/`, which holds the paper's figures and tables and nothing else. Per task: quality against training time for FUTON and the strongest model of each other family, with a within-signal standard-error band; every model's final quality against its training time and against its inference rate; one table of every model's size, training time and final metrics, the averages with a paired standard error, the best in bold and the second underlined; and, for each of two signals, that signal with two regions boxed and those regions magnified for every featured model, the regions found as the two squares of fine detail where FUTON gains most over the strongest baseline. Figures come as PDF and as PGF to include in a LaTeX document. The FUTON ablations add a table and a convergence plot of the bases, one table of the combiner and the decoder together, a convergence plot of the combiner, and IoU against the CP rank at each number of components and the other way round.

```bash
python scripts/report_paper.py                   # everything, into results/
python scripts/report_paper.py --tasks image --overwrite   # redraw one task's panels
python scripts/report_paper.py --no-panels       # tables and plots, without a GPU
python scripts/report_paper.py --orbit 60        # also an orbit GIF per NeRF model
python scripts/profile_speed.py                  # -> logs/speed/, timed in one job
```

The magnified panels are rebuilt from the checkpoints, which needs the signals in `data/` and a GPU, so they are kept in `results/<task>/panels/` and reused; `--overwrite` draws them again.

A run's recorded time carries whatever else shared its node, so
`scripts/profile_speed.py` times every model's inference on one signal per
task, in turn, for a single (ideally exclusive) job to run.

Or read the runs directly:

```python
records = [json.loads(p.read_text()) for p in Path("logs").rglob("results.json")]
table = pd.json_normalize(records)
```

## Package Structure

```
neurofield/
├── models/                   # architectures (re-exported as nf.*)
│   ├── mlp.py                # MLP, ReLULayer
│   ├── siren.py              # SIREN
│   ├── finer.py              # FINER
│   ├── gauss.py              # Gauss
│   ├── wire.py               # WIRE, RealWIRE
│   ├── rff.py                # RFF, RFFEncoding
│   ├── pemlp.py              # PEMLP, PositionalEncoding
│   ├── mfn.py                # MFN
│   ├── instant_ngp.py        # InstantNGP, HashEncoding
│   ├── multivector.py        # MultiVector, FeatureGrid
│   ├── gaplanes.py           # GAPlanes
│   ├── tensorf.py            # TensoRF
│   ├── grid_inr.py           # GridINR
│   ├── futon.py              # FUTON, bases, combiners
│   ├── rcs_matrix.py         # RCSMatrix, rcs_product (row-contiguous sparse features)
│   └── dip.py                # DIPUNet, DIPSkip
├── datasets.py               # ImageCoordinateDataset, OccupancyCoordinateDataset, MRICoordinateDataset, ...
├── losses.py                 # rate_distortion_loss, sdf_loss, soft_entropy, laplace_entropy
├── metrics.py                # psnr, ssim, ms_ssim, lpips, nmse, iou, bits_per_pixel, ...
├── quantization.py           # uniform_quantize, uniform_dequantize
├── training.py               # train, train_epoch
├── evaluation.py             # evaluate, chunked_inference
├── visualization.py          # LinePlot (seaborn-based)
├── utils.py                  # JSONLogger, encode_dict, decode_dict, radon, sobel_gradients, ...
├── compression/              # nf.compression
│   ├── codecs.py             # encode_futon, decode_futon, encode_pil, decode_pil
│   ├── evaluation.py         # evaluate_compression
│   └── arithmetic_coding.py  # encode_arithmetic, decode_arithmetic
└── nerf/                     # nf.nerf
    ├── cameras.py            # generate_ray_directions, generate_rays, pose_spherical, ...
    ├── datasets.py           # BlenderDataset
    ├── fields.py             # RadianceField, SphericalHarmonicsEncoding, IdentityEncoding
    ├── renderers.py          # create_renderer, VolumeRenderer, PyTorchRenderer, NerfAccRenderer
    ├── training.py           # train
    └── evaluation.py         # evaluate, render_image, render_views
```

## Notes

- **`MLP` supports ordinary and custom hidden layers**: it uses linear layers with ReLU by default. Both `activation` and `output_activation` accept tensor callables such as `torch.tanh` or module instances such as `torch.nn.PReLU()`. Hidden activation modules are shared across layers. Alternatively, `layer_class` supplies complete hidden layers with their own activation and constructor keyword arguments. Replace `CoordinateMLP` with `MLP`, passing `layer_class` and `output_activation` by keyword. Use `layer_class=torch.nn.Linear` and `hidden_layers=3` if you relied on the old defaults. Subclasses now apply custom initialization after `super().__init__`; SIREN and FINER already do this.
- **`MLP` checkpoints**: `layers` now contains only weight layers. For ordinary MLPs, remap old `layers.{2*i}` keys to `layers.{i}` and keep shared activation parameters under `activation.*` only. Custom-layer checkpoint keys are unchanged.
- **Shared grid decoders**: GAPlanes, TensoRF, and InstantNGP use `MLP`, as FUTON already does with `decoder="mlp"`. For older grid checkpoints, remap `decoder.{2*i}` to `decoder.layers.{i}`; other keys are unchanged.
- **`hidden_layers` counts hidden layers**: a feed-forward network with `hidden_layers=h` has `h + 1` weight layers (MFN: `h` Gabor filters, `h - 1` hidden linear layers and one output linear layer). Configurations logged before this convention for SIREN, FINER, Gauss, WIRE / RealWIRE, RFF, PEMLP and MFN need `hidden_layers + 1` to rebuild the same architecture; `MLP`, GAPlanes, TensoRF and InstantNGP are unchanged.
- **WIRE in the experiments** uses the real Gabor wavelet layer (`nf.RealWIRE`), one of the two variants in the WIRE paper and its official code. At equal parameter counts it trains about 1.5x faster than the complex layer (`nf.WIRE`), and its weights are real, so it also works with quantization.
- **Coordinates** live in `[-1, 1]`: the coordinate datasets generate them in that range, and FUTON bases, hash and feature grids, and GridINR patches are defined on it (do not pre-normalize to `[0, 1]`).
- **Metrics**: `nf.psnr`, `nf.ssim`, `nf.ms_ssim` and `nf.lpips` take `(C, H, W)` images; `data_range` defaults to 255 for `uint8` images and 1 for float images in `[0, 1]`. `nf.nerf.evaluate` reports PSNR, SSIM and LPIPS (AlexNet) by default.

## Development

Install the development tools, format the package, and run the tests:

```bash
pip install -e ".[dev]"
python -m black neurofield
python -m pytest -q
```

Use `python -m black --check neurofield` to check formatting without changing files.
Tests that require CUDA or local datasets are skipped when those are unavailable.

## License

This project is licensed under the MIT License.
