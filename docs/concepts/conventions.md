# Conventions

The rules that let datasets, models, losses and metrics be combined freely.

## Coordinates

Coordinates live in $[-1, 1]$ on every axis, with both endpoints included: `nf.create_coordinates((H, W))` returns an `(H, W, 2)` grid whose column $c$ spans axis $c$ (`indexing="ij"`), and the datasets build their inputs with it. FUTON's bases, the hash and feature grids, and `GridINR` patches are defined on this domain, so do not pre-normalize to $[0, 1]$. Models accept any leading shape; FUTON's sparse bases need a flat `(N, C)` batch, which `FUTON.forward` produces internally.

## Values

Targets are in the model's output range and the reference signal keeps its native form:

| Dataset | `input` | `target` | `_original` |
| --- | --- | --- | --- |
| `ImageCoordinateDataset` | `(H, W, 2)` | `(H, W, C)` float in $[-1, 1]$ | `(C, H, W)` `uint8` |
| `MaskedImageCoordinateDataset` | `(H, W, 2 + C_s)` with a guide image | `(H, W, C)` in $[-1, 1]$, plus `side`, `mask`, `boundary` | `(C, H, W)` `uint8` |
| `OccupancyCoordinateDataset` | `(D, H, W, 3)` | `(D, H, W, 1)` in $\{-1, 1\}$ | `(1, D, H, W)` float in $\{0, 1\}$ |
| `MRICoordinateDataset` | `(H', W', 2)` | complex `(coils, H, W)` masked k-space, plus `mask`, `scale_factor`, `zero_filled` | RSS image `(H', W')` or `None` |
| `DIPImageDataset` | noise `(C_n, H, W)` | `(C, H, W)` float in $[0, 1]$ | `(C, H, W)` `uint8` |
| `nerf.BlenderDataset` | rays `(H, W, 3)` origins and directions | `(H, W, 4)` RGBA in $[0, 1]$ | |

Subsampling (`subsample`, `max_samples`) flattens `input` and `target` to `(N, ...)` with shared random indices. `preprocess` maps native values to targets (`uint8` to $[-1, 1]$; occupancy $\{0, 1\}$ to $\{-1, 1\}$) and `postprocess` maps model output back (clipping to $[-1, 1]$ before the `uint8` cast; thresholding occupancy at zero). Because targets are in $[-1, 1]$, the benchmark models end in `torch.tanh`, and the default training loss reports PSNR with a peak-to-peak range of 2.

## Metrics

`nf.psnr`, `nf.ssim`, `nf.ms_ssim` and `nf.lpips` take two `(C, H, W)` images and infer `data_range` from the target: 255 for `uint8`, 1 for floating point. `nf.lpips` needs three channels and uses the VGG backbone by default (`net="alex"` selects AlexNet); its weights are downloaded once. `nf.iou` thresholds both inputs at zero, so it works on the `{0, 1}` original and on raw model output alike. `nf.nmse` is fastMRI's normalized MSE, and `nf.bits_per_pixel(encoded, image)` divides a byte string's size by the spatial size of a `(C, *spatial)` image. In `nf.train` and `nf.evaluate`, metrics are called as `fn(reconstruction, original)` after `postprocess`, so they see `uint8` images and `{0, 1}` volumes.

## Widths, depths and activations

`hidden_layers=h` means `h` hidden layers and `h + 1` weight layers. `output_activation` is applied after the last layer of every model; inside `nf.nerf.RadianceField` the density network must have none, because the field applies `trunc_exp` to the density and a sigmoid to the colour.

## Checkpoints

`nf.train` saves the model's state dict as `checkpoint.pt` in `log_dir` and returns it as `model_dict`. With `quantize=True`, the checkpoint holds the quantized weights and `model_dict` instead holds a dictionary of integer codes and per-tensor scales, which `nf.uniform_dequantize` loads back. `nf.nerf.train` saves the field's state dict, including the `aabb` buffer. `MLP.layers` holds only weight layers (`layers.0` to `layers.h`), and grid models keep their decoder under `decoder.layers.*`.

## Seeds and devices

`seed` in `nf.train` and `nf.nerf.train` seeds PyTorch's global RNG before training, after the model has been built; construct the model after `torch.manual_seed` to also fix its initialization, as the scripts do. A dataset's `seed` reseeds Python's RNG only; subsampling draws from PyTorch's RNG. `device=None` selects CUDA when available. The training loop moves the model and, through their `to` method, the datasets to that device, and `_original` stays on the CPU.

## Experiment files

A benchmark run lives in `logs/<task>/<signal>/<model>/` with `log.txt`, `log.json`, `checkpoint.pt`, the final reconstruction (or two rendered test views for NeRF), and `results.json`: the training configuration, the model's `setup` from the YAML, `num_params`, `train_time`, the final `metrics`, and the `history`. Failed runs leave `error.txt`. The [reproduction page](../experiments/reproduce.md) describes the scripts that write and read them.
