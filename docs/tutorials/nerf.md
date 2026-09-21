# Radiance fields

`nf.nerf` turns any model of the zoo into the density network of a radiance field, trains it on posed images by volume rendering, and scores held-out views. The protocol is FINER's, which is torch-ngp's Blender recipe: 200 × 200 views, 25 training views, 37,500 steps, scored on all 200 test views. This follows `configs/nerf.yaml` and `scripts/train_nerf.py`.

<figure markdown="span">
  ![Orbit around the lego scene](../assets/orbit_lego.gif){ width="720" }
  <figcaption>The lego scene rendered in orbit by SIREN, Instant-NGP and FUTON-sinc at equal parameter count (about 75k), with the test-view PSNR.</figcaption>
</figure>

## Data

```python
import neurofield as nf

root = "data/nerf/blender/lego"
train_set = nf.nerf.BlenderDataset(root, split="train", downsample=4, skip=4)
val_set = nf.nerf.BlenderDataset(root, split="val", downsample=4)
test_set = nf.nerf.BlenderDataset(root, split="test", downsample=4)
```

`downsample=4` reduces the 800 × 800 renders to 200 × 200 by area averaging; `skip=4` keeps 25 of the 100 training views. Camera positions are scaled by `scene_scale=0.8` into the box `[-bound, bound]^3` with `bound=1`, marching starts at `near=0.2`, and the background is white. `dataset[i]` returns a full view's rays and RGBA; `sample_rays` draws random rays across views for training without materializing a ray buffer.

## Field and renderer

```python
field = nf.nerf.RadianceField(
    density_net=(nf.FUTON, {
        "basis": ("lanczos", {"num_components": 128, "radius": 3}),
        "combiner": ("cp", {"rank": 128}),
        "decoder": ("mlp", {"hidden_layers": 1}),
    }),
    color_net=("mlp", {"hidden_features": 64, "hidden_layers": 2}),
    geometry_features=15,
    direction_encoding="sh",
    aabb=train_set.aabb,
)
renderer = nf.nerf.create_renderer(aabb=train_set.aabb, near=train_set.near, far=train_set.far)
```

The density network receives `in_features=3` and `out_features=1 + geometry_features`: one density logit and 15 features, on positions mapped from the scene box to $[-1, 1]^3$. The colour network maps those features and the encoded view direction (`"sh"`: 16 real spherical harmonics, as in Instant-NGP; `"identity"`: raw directions, as in FINER) to RGB. The field applies `trunc_exp` to the density and a sigmoid to the colour, so neither network has an output activation. Every model in the benchmark shares this colour network and renderer; only the density network changes, so the comparison is between spatial representations.

`create_renderer` picks nerfacc when its CUDA extension works and the pure-PyTorch renderer otherwise; both march rays through an occupancy grid (`grid_resolution=128`, updated every 16 steps) with a step of $\sqrt{3} \cdot \text{extent} / 1024$ and at most `max_samples_per_ray=1024` occupied samples.

## Training

```python
result = nf.nerf.train(
    field,
    renderer,
    train_set,
    eval_dataset=val_set,
    eval_indices=[0, 25, 50, 75],
    num_steps=37500,
    num_rays=4096,
    lr=3e-2,
    eval_interval=2500,
    log_dir="logs/tutorial/lego/FUTON-lanczos",
)
```

Each step updates the occupancy grid, renders 4096 random training rays, composites predictions and targets onto a random background colour, and takes an Adam step (`adam_betas=(0.9, 0.99)`, `adam_eps=1e-15`) with the learning rate decaying exponentially to `lr / 10`. `history` records `step`, `elapsed`, `loss`, the training `psnr` and `num_samples` every `log_interval` steps, and the mean validation metrics under `"eval"` every `eval_interval`. Learning rates in `configs/nerf.yaml` range from 3e-4 (SIREN, FINER) to 3e-2 (FUTON, TensoRF, GA-Planes); a run of 37,500 steps takes FUTON about six minutes on an A100.

## Evaluation and rendering

```python
scores = nf.nerf.evaluate(field, renderer, test_set)        # PSNR, SSIM and LPIPS on all 200 views
print(scores["mean"])
```

`scores["views"]` holds one record per view with its index, render time and metrics; `return_images=True` adds the rendered `(H, W, 3)` images; `indices` selects a subset. `metrics` accepts any `fn(pred, target)` on `(3, H, W)` images in $[0, 1]$.

To render new viewpoints, build camera-to-world poses and call `render_views`:

```python
import imageio.v2 as imageio

poses = [nf.nerf.pose_spherical(theta, -30.0, train_set.radius) for theta in range(0, 360, 12)]
frames = nf.nerf.render_views(field, renderer, test_set, poses)   # list of (H, W, 3) uint8 arrays
imageio.mimsave("lego_orbit.gif", frames, duration=0.083, loop=0)
```

`render_image(field, renderer, rays_o, rays_d)` renders arbitrary rays, returning RGB, opacity and depth.

## The benchmark script

`scripts/train_nerf.py` trains every model of the config on every scene under `data/nerf/blender/`, evaluates on the 200 test views, and saves two rendered test views per run. The NeRF sweep is the long one: about ten minutes per model and scene on an A100, 104 runs in all.

```bash
python scripts/train_nerf.py --data data/nerf/blender/lego --models FUTON-sinc
python scripts/report_paper.py --tasks nerf --orbit 60    # figures, tables and an orbit GIF per model
```
