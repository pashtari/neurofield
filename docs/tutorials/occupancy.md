# Occupancy volumes

A shape is represented as an occupancy field $[-1, 1]^3 \to \{-1, 1\}$, fitted to a voxelized mesh and scored by intersection over union. This follows `configs/occupancy.yaml` and `scripts/train_occupancy.py`.

## Data

`OccupancyCoordinateDataset` loads a mesh (any format trimesh reads, with the `3d` extra) or an `.xyz` point cloud with normals, normalizes it into $[-0.45, 0.45]^3$ as FINER and BACON do, voxelizes it at `resolution` samples per unit length, crops the volume to its occupied bounds, and caches the result beside the source as `.cache/<name>_<resolution>.pt`.

```python
import torch
import neurofield as nf

path = "data/occupancy/thai_statue.ply"
train_set = nf.OccupancyCoordinateDataset(path, resolution=256, subsample=0.01)
eval_set = nf.OccupancyCoordinateDataset(path, resolution=256)

eval_set.input.shape, eval_set.target.shape   # (D, H, W, 3) coordinates, (D, H, W, 1) targets in {-1, 1}
eval_set.original.shape                        # (1, D, H, W) float occupancy in {0, 1}
```

At $256^3$ resolution a volume has up to 16.8 million voxels; `subsample=0.01` trains on a random 1 % of them per step. The five meshes of the benchmark are the Stanford armadillo, dragon, happy buddha, lucy and thai statue, fetched by `scripts/download_meshes.py`.

## Model

The benchmark FUTON uses 128 components per axis, CP rank 218 and a one-layer MLP decoder, 131,673 parameters:

```python
model = nf.FUTON(
    3, 1,
    basis=("sinc", {"num_components": 128, "grid_size": 256}),
    combiner=("cp", {"rank": 218}),
    decoder=("mlp", {"hidden_layers": 1}),
    output_activation=torch.tanh,
)
```

The Lanczos variant replaces the basis with `("lanczos", {"num_components": 128, "radius": 3})`; it needs no `grid_size` because sparse mode evaluates its six taps directly. `grid_size=256` tabulates the sinc basis on 256 points per axis; coordinates that fall on them are looked up and the rest are evaluated directly, so it is safe whatever size the cropped volume has.

## Training and evaluation

```python
result = nf.train(
    model,
    train_set,
    eval_set,
    num_epochs=2000,
    lr=1e-2,
    metrics={"iou": nf.iou},
    chunk_size=262144,
    log_interval=100,
    eval_interval=100,
    save_reconstruction=False,
    log_dir="logs/tutorial/thai_statue/FUTON-sinc",
)
print(result["history"][-1]["eval"]["iou"])
```

`chunk_size=262144` bounds the memory of evaluating the whole volume. `nf.iou` thresholds the model's output at zero and compares it with the `{0, 1}` original. The benchmark models reach 99.5 to 99.9 % IoU in 2000 epochs; FUTON's convergence is shown on the [benchmarks page](../experiments/benchmarks.md#occupancy).

## Exporting a mesh

With the `3d` extra, `eval_set.save(volume, path)` smooths the occupancy, runs marching cubes, writes a `.dae` mesh and renders it to a `.png` through a hidden Open3D window:

```python
volume = nf.chunked_inference(model, eval_set.input, chunk_size=262144)
occupancy = eval_set.postprocess(volume.cpu())       # (1, D, H, W) in {0, 1}
eval_set.save(occupancy, "thai_statue_futon")         # thai_statue_futon.dae and .png
```

The render needs a display, which compute nodes lack; that is why the call above and the benchmark script pass `save_reconstruction=False` and rebuild meshes from `checkpoint.pt` afterwards (`nf.evaluate(model, eval_set, ckpt_path=..., log_dir=...)` on a workstation).

## The benchmark script

```bash
python scripts/train_occupancy.py                                       # all shapes, all models
python scripts/train_occupancy.py --data data/occupancy/lucy.ply --models FUTON-lanczos
python scripts/train_occupancy.py --config configs/ablation-futon/basis.yaml   # -> logs/ablation-futon/basis/
```

The four FUTON ablation configs in `configs/ablation-futon/` run on this task; the [ablations page](../experiments/ablations.md) reports them.
