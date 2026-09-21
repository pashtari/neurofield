# Troubleshooting

The problems that come up first, and what to do about them.

## Coordinates and targets

**The fit is poor or the loss does not move.**
Coordinates must lie in $[-1, 1]$ on every axis. The coordinate datasets generate them that way, and FUTON's bases, the hash and feature grids, and `GridINR`'s patches are all defined on that domain. Do not pre-normalize to $[0, 1]$. Targets from the datasets are in $[-1, 1]$ too (occupancy in $\{-1, 1\}$), which is why the benchmark models end in `output_activation=torch.tanh` and why the default loss reports PSNR for that range.

**`ValueError: num_components must be >= ...`**
`SincBasis` needs at least two centres per axis, and a local basis needs at least `2 * radius`. With `num_components=[H // 2, W // 2]` this is never an issue for real images; it is for tiny test inputs.

**Gradients with respect to the coordinates are zero.**
A basis built with `grid_size` reads on-grid coordinates from a table, which carries no coordinate gradient. Leave `grid_size=None` when the loss differentiates through the input, for example an eikonal term.

**`grid_size` on a dataset and `subsample` together.**
`ImageCoordinateDataset(image, grid_size=...)` evaluates a model on another grid than the image's (for example a finer one) and cannot subsample; it is for evaluation.

## Memory

**CUDA out of memory during evaluation.**
Pass `chunk_size` to `nf.train`, `nf.evaluate` or `nf.chunked_inference`; the occupancy benchmark evaluates $256^3$ volumes with `chunk_size=262144`. In `nf.nerf.train`, lower `num_rays` or `max_samples_per_ray`; in `nf.nerf.evaluate` and `render_image`, pass `chunk_size` (rays per pass).

**CUDA out of memory during training.**
Lower `subsample` (the fraction of coordinates per step); the benchmarks use 0.1 for images and 0.01 for volumes.

## Backends

**`NerfAcc: No CUDA toolkit found. NerfAcc will be disabled.`**
nerfacc builds its CUDA extension on first use and needs `nvcc` for that. Without it, `nf.nerf.create_renderer()` selects the PyTorch renderer, which produces the same renders more slowly; training still works. Install a CUDA toolkit matching your PyTorch build to use nerfacc.

**`UserWarning: Sparse CSR tensor support is in beta state`.**
Raised by PyTorch when a local basis runs on the CPU or in inference mode, where sparse products use CSR. It is harmless.

**No Triton.**
The fused kernels for local bases need Triton, which ships with PyTorch's Linux CUDA wheels. Elsewhere the same products are computed with gathers and contractions, with identical results.

## Data and extras

**`ModuleNotFoundError: trimesh` (or `mcubes`, `open3d`).**
Meshes as occupancy volumes, mesh export and offscreen rendering need the `3d` extra: `pip install -e ".[3d]"`. Point clouds in `.xyz` format load without it.

**Rendering a mesh hangs or fails on a compute node.**
`OccupancyCoordinateDataset.save` renders through a hidden Open3D window, which needs a display. Train with `save_reconstruction=False` (as `scripts/train_occupancy.py` does) and rebuild the mesh from `checkpoint.pt` on a machine with a display.

**LPIPS downloads weights on first use.**
`nf.lpips` loads the VGG (or AlexNet) backbone through torchmetrics, which fetches the weights once. On nodes without internet, run one evaluation on the login node first, with the same cache directory.

## Experiments

**A run is skipped with "holds a run of a different setup".**
Runs are keyed by model name inside the log directory. After changing a model's entry in a config, pass `--overwrite` to rerun it in place, or give the variant its own name or `--log-dir`.

**A model diverges.**
Learning rates in the configs come from the grid {3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4}, and the comments beside them record what the neighbours cost. Sine networks (SIREN, FINER) are the most sensitive; FUTON tolerates 1e-2 to 3e-2 across tasks.

**Training times differ from the tables.**
The recorded times were measured on one A100 in jobs that shared their nodes; the [reproduction page](experiments/reproduce.md) explains how `scripts/profile_speed.py` times inference in one exclusive job.

## Models

**`hidden_layers` counts hidden layers.**
A network with `hidden_layers=h` has `h + 1` weight layers (MFN: `h` Gabor filters and `h` linear layers). The configs state the counts as the constructors expect them.

**Loading an `MLP` checkpoint.**
`MLP.layers` contains only weight layers, so a network with `hidden_layers=h` has keys `layers.0` to `layers.h`; a shared activation with parameters (for example `nn.PReLU`) lives under `activation.*`.

**Quantized checkpoints.**
With `nf.train(..., quantize=True)`, `checkpoint.pt` holds the quantized weights as an ordinary state dict, while the returned `model_dict` holds integer codes with per-tensor scales; restore those with `nf.uniform_dequantize(model, model_dict)`.
