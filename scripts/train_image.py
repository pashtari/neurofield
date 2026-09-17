"""Image representation benchmark: fit each model in configs/image.yaml to Kodak images.

Usage:
    python scripts/train_image.py
    python scripts/train_image.py --data data/Kodak/kodim01.png --models SIREN FINER --device cuda:1
"""

import re
from pathlib import Path
from typing import Any

import torch
from PIL import Image

import neurofield as nf
from train_common import ROOT, build, record, run

DATA = sorted((ROOT / "data" / "Kodak").glob("kodim*.png"))


def resolve(value: Any, **sizes: int) -> Any:
    """Evaluate size expressions such as ``"max(H, W) // 2"``, recursively."""
    if isinstance(value, dict):
        return {key: resolve(item, **sizes) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve(item, **sizes) for item in value]
    if isinstance(value, str) and re.fullmatch(r"[HW\d\s()+\-*/,max]+", value):
        return eval(value, {"__builtins__": {}, "max": max}, sizes)
    return value


def prepare(model: dict, path: Path) -> dict[str, Any]:
    """Fill in the image height H and width W of the model's settings."""
    width, height = Image.open(path).size
    return resolve(model, H=height, W=width)


def fit(
    config: dict, model: dict, path: Path, out: Path, device: torch.device
) -> dict[str, Any]:
    model_class, kwargs = build(model)
    res = nf.train(
        model_class(in_features=2, out_features=3, **kwargs),
        nf.ImageCoordinateDataset(path, **config["data"]),
        nf.ImageCoordinateDataset(path),
        metrics={
            "psnr": nf.psnr,
            "ssim": nf.ssim,
            "ms_ssim": nf.ms_ssim,
            "lpips": nf.lpips,
        },
        device=device,
        log_dir=out,
        **{**config["train"], **model["train"]},
    )
    return record(res, model, res["history"][-1]["eval"])


if __name__ == "__main__":
    run("image", DATA, fit, prepare)
