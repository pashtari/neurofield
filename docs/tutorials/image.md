# Fitting an image

The image task fits a field $[-1, 1]^2 \to \mathbb{R}^3$ to one image and scores it on every pixel. This tutorial follows `configs/image.yaml` and `scripts/train_image.py`, so what you run here is what the [benchmark](../experiments/benchmarks.md#images) ran.

## Data

```python
import torch
import neurofield as nf

path = "data/Kodak/kodim19.png"
train_set = nf.ImageCoordinateDataset(path, subsample=0.1)
eval_set = nf.ImageCoordinateDataset(path)
H, W = eval_set.grid_size
```

`subsample=0.1` draws a fresh random tenth of the pixels on every access, which is one training step; `max_samples` caps the draw for very large images. `mode="L"` loads a grayscale image (`out_features=1`). A tensor or PIL image can be passed instead of a path, with `item_id` naming it in logs.

## Models at equal size

Each benchmark model holds about 195k parameters. FUTON's basis size follows the image: half the sampling resolution per axis, tabulated on the pixel grid.

```python
futon = nf.FUTON(
    2, 3,
    basis=("sinc", {"num_components": [H // 2, W // 2], "grid_size": [H, W]}),
    combiner=("cp", {"rank": 224}),
    decoder=("mlp", {"hidden_layers": 1}),
    output_activation=torch.tanh,
)
futon_lanczos = nf.FUTON(
    2, 3,
    basis=("lanczos", {"num_components": [H // 2, W // 2], "radius": 3}),
    combiner=("cp", {"rank": 224}),
    decoder=("mlp", {"hidden_layers": 1}),
    output_activation=torch.tanh,
)
siren = nf.SIREN(2, 3, hidden_features=256, hidden_layers=4, omega=30.0,
                 output_activation=torch.tanh)
finer = nf.FINER(2, 3, hidden_features=256, hidden_layers=4, omega=30.0,
                 output_activation=torch.tanh)
ngp = nf.InstantNGP(
    2, 3,
    num_levels=19, features_per_level=2, log2_hashmap_size=13,
    base_resolution=16, max_resolution=max(H, W) // 2,
    hidden_features=64, hidden_layers=2, output_activation=torch.tanh,
)
tensorf = nf.TensoRF(2, 3, rank=176, resolution=256, mode="cp",
                     hidden_features=256, hidden_layers=2, output_activation=torch.tanh)
```

`configs/image.yaml` lists all twelve, with their learning rates and, where a setting departs from the authors' code, a comment saying why. The learning rate matters: FUTON trains at 3e-2, SIREN and FINER at 1e-3 (3e-3 diverges for SIREN), Instant-NGP at 1e-2 with `adam_betas=(0.9, 0.99)` and `adam_eps=1e-15`. Use `nf.count_parameters(model, trainable_only=False)` when you change a size.

## Training

```python
result = nf.train(
    futon,
    train_set,
    eval_set,
    num_epochs=2000,
    lr=3e-2,
    metrics={"psnr": nf.psnr, "ssim": nf.ssim, "ms_ssim": nf.ms_ssim, "lpips": nf.lpips},
    log_interval=100,
    eval_interval=100,
    log_dir="logs/tutorial/kodim19/FUTON-sinc",
)
```

The metrics run on the `uint8` reconstruction against the original every 100 epochs. LPIPS is the slow one (it runs a VGG network each time); leave it out of `metrics` while iterating and compute it once at the end with `nf.evaluate`.

## Learning curves

`history` is a list of records, one per logged epoch, with the evaluation metrics under `"eval"`:

```python
import pandas as pd

curve = pd.DataFrame(
    {"epoch": h["epoch"], "time": h["elapsed"], **h["eval"]}
    for h in result["history"] if "eval" in h
)
curve.plot(x="time", y="psnr")
```

`nf.LinePlot` wraps seaborn for the multi-model case: collect one record per model and epoch and call `LinePlot(records).plot(x="time", y="psnr", hue="model")`. The notebook `notebooks/image_representation.ipynb` does exactly this for all twelve models.

## Reconstruction

```python
output = nf.chunked_inference(futon, eval_set.input, chunk_size=65536)  # (H, W, 3)
image = eval_set.postprocess(output.cpu())                               # uint8 (3, H, W)
eval_set.save(image, "kodim19_futon.png")
```

`nf.evaluate(model, eval_set, metrics=..., ckpt_path="logs/.../checkpoint.pt", log_dir=...)` does the same from a checkpoint and writes the reconstruction beside the logs.

## Masks and guide images

`MaskedImageCoordinateDataset(image, side=..., mask=...)` appends a preprocessed guide image to the coordinates and returns a boolean `mask` and its thick `boundary` with every item, at the image resolution. The default loss ignores them; a custom loss that fits only the observed pixels is a few lines, see [custom components](custom-components.md#a-loss).

## The benchmark script

`scripts/train_image.py` runs every model of the config on every image under `data/Kodak/`, writes each run to `logs/image/<image>/<model>/`, and skips runs that already have a `results.json`:

```bash
python scripts/train_image.py                                        # all images, all models
python scripts/train_image.py --data data/Kodak/kodim19.png --models FUTON-sinc SIREN
python scripts/train_image.py --set train.num_epochs=500 --log-dir logs/short
```

Sizes in the config may be expressions in the image height `H` and width `W`, such as `max(H, W) // 2`. See [Reproducing the paper](../experiments/reproduce.md) for the full command-line interface.
