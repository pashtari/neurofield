"""Super-resolution benchmark: recover DIV2K images from their 4x bicubic downsamplings.

Every model in configs/super_resolution.yaml sees only the low-resolution (LR)
image and is scored against the high-resolution (HR) original. The benchmark's
degradation, antialiased bicubic downsampling (MATLAB's imresize), is linear
and separable: one matrix per axis describes it exactly, borders included, and
an LR pixel needs the field only under its kernel. Each step renders the HR
pixels under random blocks of LR pixels, downsamples them with those matrices
and matches the LR pixels (L1). DIP renders the whole image from fixed noise,
and the interpolations upsample the LR image with PIL, whose bicubic filter is
MATLAB's. As is standard, PSNR and SSIM are computed on luma (BT.601 Y) and
LPIPS on RGB, all without a 4-pixel border.

Usage:
    python scripts/train_super_resolution.py
    python scripts/train_super_resolution.py --data data/DIV2K/DIV2K_valid_HR/0882.png --models Bicubic FUTON-sinc
"""

import math
import time
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn
from torchvision.transforms.functional import pil_to_tensor
from train_common import ROOT, build, fresh, on_grid, record, resolve, run, warm_up

import neurofield as nf

DATA = sorted((ROOT / "data" / "DIV2K" / "DIV2K_valid_HR").glob("*.png"))
INTERPOLATIONS = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
}


def low_resolution(path: Path, scale: int) -> Path:
    """The DIV2K release's bicubic downsampling of a high-resolution image."""
    folder = path.parent.name.replace("HR", "LR_bicubic")
    return path.parents[1] / folder / f"X{scale}" / f"{path.stem}x{scale}{path.suffix}"


def resize_matrix(size_in: int, size_out: int, device: torch.device) -> Tensor:
    """The ``(size_out, size_in)`` matrix of antialiased bicubic resizing along one axis.

    Resizing the rows of an identity image while keeping its columns writes
    the matrix of ``F.interpolate`` out, borders included.
    """
    identity = torch.eye(size_in, device=device)[None, None]
    size = (size_out, size_in)
    return F.interpolate(identity, size=size, mode="bicubic", antialias=True)[0, 0]


def downsample(image: Tensor, kernels: Sequence[Tensor]) -> Tensor:
    """Apply one kernel per axis: ``(..., H, W, C)`` to ``(..., h, w, C)``."""
    return torch.einsum("...il,...lmc,...jm->...ijc", kernels[0], image, kernels[1])


class SuperResolutionDataset(nf.ImageCoordinateDataset):
    """LR pixels as targets, the HR coordinates under their kernels as inputs.

    Each access draws random ``block x block`` groups of LR pixels that cover
    a fraction ``subsample`` of the LR image. Neighbouring LR pixels share most
    of their footprint, so a block needs far fewer field evaluations per pixel
    than single pixels do: 30 in an 8 x 8 block against 256 alone. Items hold
    ``input``, the HR coordinates of each block's footprint, of shape
    ``(B, L, L, 2)``; ``weight_y`` and ``weight_x``, the kernel rows of each
    block on its footprint, of shape ``(B, block, L)``; and ``target``, the LR
    pixels in ``[-1, 1]``, of shape ``(B, block, block, C)``. Every tensor
    lives on the kernels' device.
    """

    def __init__(
        self, image: Path, kernels: Sequence[Tensor], subsample: float, block: int
    ) -> None:
        device = kernels[0].device
        self.original = self.load(image, mode="RGB")  # uint8 (C, h, w)
        self.id = image.stem
        self.kernels, self.subsample, self.block = kernels, subsample, block
        self.grid_size = tuple(kernel.shape[1] for kernel in kernels)  # (H, W)
        self.input = self.create_input().to(device)  # HR coordinates, (H, W, 2)
        self.target = self.preprocess(self.original).to(device)  # (h, w, C)

        # The first tap of each kernel row, and the footprint of a block along one axis.
        self.first = [(kernel != 0).int().argmax(1) for kernel in kernels]
        taps = max(int((kernel != 0).sum(1).max()) for kernel in kernels)
        scale = self.grid_size[0] // self.target.shape[0]
        self.length = scale * (block - 1) + taps

    def __getitem__(self, index: int) -> dict[str, Any]:
        device = self.target.device
        h, w = self.target.shape[:2]
        count = max(1, round(self.subsample * h * w / self.block**2))
        offsets = torch.arange(self.block, device=device)
        span = torch.arange(self.length, device=device)

        pixels, taps, weights = [], [], []
        for size, full, kernel, first in zip(
            (h, w), self.grid_size, self.kernels, self.first
        ):
            start = torch.randint(size - self.block + 1, (count, 1), device=device)
            pixel = start + offsets  # (B, b)
            tap = first[start].clamp(max=full - self.length) + span  # (B, L)
            weights.append(kernel[pixel[:, :, None], tap[:, None, :]])  # (B, b, L)
            pixels.append(pixel)
            taps.append(tap)

        return {
            "id": self.id,
            "input": self.input[taps[0][:, :, None], taps[1][:, None, :]],
            "weight_y": weights[0],
            "weight_x": weights[1],
            "target": self.target[pixels[0][:, :, None], pixels[1][:, None, :]],
            "_original": self.original,
        }


def l1_loss(output: Tensor, target: Tensor, peak: float) -> tuple[Tensor, dict]:
    """Mean absolute error, with the PSNR of the fit for the training log."""
    loss = F.l1_loss(output, target)
    mse = F.mse_loss(output, target).item()
    return loss, {"l1": loss.item(), "psnr": 10 * math.log10(peak**2 / mse)}


def field_loss(batch: dict[str, Any], model: nn.Module) -> tuple[Tensor, dict]:
    """Render each footprint, downsample it along both axes, and match the LR pixels."""
    weights = batch["weight_y"], batch["weight_x"]
    return l1_loss(downsample(model(batch["input"]), weights), batch["target"], 2.0)


def prior_loss(
    batch: dict[str, Any], model: nn.Module, kernels: Sequence[Tensor]
) -> tuple[Tensor, dict]:
    """Render the HR image from noise, downsample it, and match the LR image."""
    output = downsample(model(batch["input"]).movedim(1, -1), kernels)
    return l1_loss(output, batch["target"].movedim(1, -1), 1.0)


def luma(image: Tensor) -> Tensor:
    """BT.601 luma of a uint8 RGB ``(3, H, W)`` image, as MATLAB's rgb2ycbcr."""
    weights = torch.tensor([65.481, 128.553, 24.966], device=image.device) / 255
    return 16 + torch.einsum("chw,c->hw", image.float(), weights).unsqueeze(0)


def metrics(device: torch.device, border: int) -> dict[str, Callable]:
    """PSNR and SSIM of luma and LPIPS of RGB, without the border, on ``device``."""

    def shave(image: Tensor) -> Tensor:
        return image.to(device)[:, border:-border, border:-border]

    def of_luma(metric: Callable[..., float]) -> Callable[[Tensor, Tensor], float]:
        return lambda pred, target: metric(
            luma(shave(pred)), luma(shave(target)), data_range=255
        )

    return {
        "psnr": of_luma(nf.psnr),
        "ssim": of_luma(nf.ssim),
        "lpips": lambda pred, target: nf.lpips(shave(pred), shave(target)),
    }


def interpolate(
    model: dict, path: Path, scale: int, device: torch.device
) -> dict[str, Any]:
    """Upsample the LR image with PIL, scored like the trained models."""
    original = nf.ImageCoordinateDataset.load(path, mode="RGB")
    with Image.open(low_resolution(path, scale)) as image:
        start = time.perf_counter()
        upsampled = image.convert("RGB").resize(
            (original.shape[2], original.shape[1]),
            INTERPOLATIONS[model["interpolation"]],
        )
        duration = time.perf_counter() - start
    reconstruction = pil_to_tensor(upsampled)
    evaluation = {"id": path.stem, "elapsed": duration, "duration": duration}
    evaluation |= {
        name: metric(reconstruction, original)
        for name, metric in metrics(device, scale).items()
    }
    history = [{"epoch": 0, "elapsed": 0.0, "duration": 0.0, "eval": evaluation}]
    return record({"config": {"num_params": 0}, "history": history}, model, evaluation)


def futon_rank(h: int, w: int, budget: int) -> int:
    """The largest CP rank at which FUTON has at most ``budget`` parameters.

    Rank ``R`` costs ``R * (h + w)`` factor entries, ``R * R + R`` for the
    hidden layer and ``3 * R + 3`` for the output layer.
    """
    b = h + w + 4
    return int((math.sqrt(b * b + 4 * (budget - 3)) - b) / 2)


def prepare(model: dict, path: Path) -> dict[str, Any]:
    """Fill in the HR height H and width W, and FUTON's rank from its budget."""
    width, height = Image.open(path).size
    model = resolve(model, H=height, W=width)
    if "budget" in model:
        name, options = model["kwargs"]["basis"]
        combiner, settings = model["kwargs"]["combiner"]
        if settings.get("rank") == "budget":
            rank = futon_rank(*options["num_components"], model["budget"])
            model["kwargs"]["combiner"] = [combiner, {**settings, "rank": rank}]
    return model


def fit(
    config: dict, model: dict, path: Path, out: Path, device: torch.device
) -> dict[str, Any]:
    scale = config["data"]["scale"]
    if "interpolation" in model:
        return interpolate(model, path, scale, device)
    model_class, kwargs = build(model)
    train = {**config["train"], **model["train"]}
    eval_dataset = nf.ImageCoordinateDataset(path)
    height, width = eval_dataset.grid_size
    kernels = [
        resize_matrix(height, height // scale, device),
        resize_matrix(width, width // scale, device),
    ]
    search = None
    if model_class is nf.DIPSkip:
        network = model_class(**kwargs)
        eval_dataset = nf.DIPImageDataset(
            path, noise_channels=kwargs["in_channels"], reg_noise_std=0
        )
        # The same seeded noise, at the HR size, perturbed at each step.
        train_dataset = nf.DIPImageDataset(
            low_resolution(path, scale), noise_shape=eval_dataset.input.shape
        )
        loss_fn = partial(prior_loss, kernels=kernels)
        train["chunk_size"] = None  # a CNN renders the whole image at once
        network.to(device).train()  # the warm-up pass, for a CNN's batched input
        network(train_dataset.input[None].to(device)).float().sum().backward()
        network.zero_grad(set_to_none=True)
    else:
        # FreSh sees what the field is fitted to: the low-resolution image.
        observed = nf.ImageCoordinateDataset(low_resolution(path, scale))
        search = fresh(
            model,
            kwargs,
            [observed.target.to(device)],
            lambda kw: on_grid(model_class(2, 3, **kw), observed, device),
            train["seed"],
        )
        network = model_class(in_features=2, out_features=3, **kwargs)
        train_dataset = SuperResolutionDataset(
            low_resolution(path, scale),
            kernels,
            config["data"]["subsample"],
            config["data"]["block"],
        )
        loss_fn = field_loss
        warm_up(network, train_dataset, device)
    res = nf.train(
        network,
        train_dataset,
        eval_dataset,
        loss_fn=loss_fn,
        metrics=metrics(device, scale),
        # Megapixel reconstructions would crowd the logs; the report renders
        # its examples from checkpoint.pt instead.
        save_reconstruction=False,
        device=device,
        log_dir=out,
        **train,
    )
    return record(res, model, res["history"][-1]["eval"], search)


if __name__ == "__main__":
    run("super_resolution", DATA, fit, prepare)
