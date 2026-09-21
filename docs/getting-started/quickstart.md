# Your first neural field

This page fits FUTON to one image, reads what the training loop returns, and swaps in another model. It needs an image; the Kodak images from the [installation page](installation.md#benchmark-data) are used below, but any RGB image works.

## Fit an image

```python
import torch
import neurofield as nf

image = "data/Kodak/kodim19.png"
dataset = nf.ImageCoordinateDataset(image)                  # every pixel, for evaluation
train_set = nf.ImageCoordinateDataset(image, subsample=0.1)  # 10 % of the pixels per step

model = nf.FUTON(
    in_features=2,
    out_features=3,
    basis=("sinc", {"num_components": 256, "grid_size": dataset.grid_size}),
    combiner=("cp", {"rank": 224}),
    decoder=("mlp", {"hidden_layers": 1}),
    output_activation=torch.tanh,
)

result = nf.train(
    model,
    train_set,
    dataset,
    num_epochs=2000,
    lr=3e-2,
    metrics={"psnr": nf.psnr, "ssim": nf.ssim},
    log_interval=500,
    eval_interval=500,
)
print(result["history"][-1]["eval"])
```

On a GPU this takes about ten seconds and ends near 37 dB PSNR. Step by step:

1. **The dataset turns the image into coordinate/value pairs.** `dataset.input` holds the pixel-centre coordinates, shape `(H, W, 2)` in $[-1, 1]$; `dataset.target` the RGB values, shape `(H, W, 3)`, also in $[-1, 1]$. With `subsample=0.1` each access draws a fresh random 10 % of the pixels, flattened to `(N, 2)` and `(N, 3)`.
2. **The model is a function of coordinates.** `FUTON` is assembled from three specs, each a registry key, a `(key, kwargs)` pair, a class, or a module instance. The sinc basis here has 256 functions per axis, tabulated on the image grid (`grid_size`) so on-grid coordinates are looked up rather than recomputed; the CP combiner has rank 224; the decoder is an MLP with one hidden layer of width 224. `output_activation=torch.tanh` matches the target range.
3. **`nf.train` runs Adam with cosine annealing** from `lr` down to `lr / 100`, one optimizer step per epoch (the dataset has one item, and the batch size is one). Every `eval_interval` epochs it evaluates on `dataset`, mapping predictions back to `uint8` with `dataset.postprocess` before computing the metrics.

The result is a plain dictionary:

```python
result["config"]       # seed, device, model, num_params, optimizer settings, ...
result["model_dict"]   # the final state dict
result["history"][-1]  # {"epoch": 2000, "elapsed": ..., "mse": ..., "psnr": ...,
                       #  "eval": {"psnr": ..., "ssim": ...}}
```

## Reconstruct and save

The trained model can be evaluated anywhere. To rebuild the image, evaluate it on the dataset's coordinate grid in chunks and map the output back to pixels:

```python
output = nf.chunked_inference(model, dataset.input, chunk_size=65536)  # (H, W, 3) in [-1, 1]
reconstruction = dataset.postprocess(output.cpu())                     # uint8 (3, H, W)

print(nf.psnr(reconstruction, dataset.original))
dataset.save(reconstruction, "kodim19_futon.png")
```

Coordinates need not lie on the pixel grid. `nf.create_coordinates((2 * H, 2 * W))` gives a twice-finer grid on the same domain, and `model(coords)` evaluates the continuous representation there; how well it interpolates depends on the basis and on `num_components`, see [FUTON](../concepts/futon.md).

## Log to disk

Pass `log_dir` to keep a record of the run:

```python
result = nf.train(model, train_set, dataset, num_epochs=2000, lr=3e-2, log_dir="logs/quickstart")
```

This writes `log.txt`, `log.json` (one JSON record per logged epoch, read back with `nf.read_json_records`), the final `checkpoint.pt`, and the evaluation reconstruction `kodim19_reconstructed.png`.

## Swap the model

Every model in the [zoo](../concepts/models.md) takes `in_features` and `out_features` first and maps `(..., in_features)` coordinates in $[-1, 1]$ to `(..., out_features)` values, so the training call does not change:

```python
model = nf.SIREN(2, 3, hidden_features=256, hidden_layers=4, output_activation=torch.tanh)
result = nf.train(model, train_set, dataset, num_epochs=2000, lr=1e-3, metrics={"psnr": nf.psnr})
```

The learning rates that the benchmark uses for each model are in `configs/image.yaml`; the [image tutorial](../tutorials/image.md) walks through them. The same loop fits occupancy volumes and, through `nf.nerf`, radiance fields; the [tutorials](../tutorials/index.md) cover each.
