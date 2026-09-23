"""Image representation benchmark: fit each model in configs/image.yaml to Kodak images.

Usage:
    python scripts/train_image.py
    python scripts/train_image.py --data data/Kodak/kodim01.png --models SIREN FINER --device cuda:1
"""

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from train_common import ROOT, build, record, resolve, run, warm_up

import neurofield as nf

DATA = sorted((ROOT / "data" / "Kodak").glob("kodim*.png"))


def prepare(model: dict, path: Path) -> dict[str, Any]:
    """Fill in the image height H and width W of the model's settings."""
    width, height = Image.open(path).size
    return resolve(model, H=height, W=width)


def fit(
    config: dict, model: dict, path: Path, out: Path, device: torch.device
) -> dict[str, Any]:
    model_class, kwargs = build(model)
    net = model_class(in_features=2, out_features=3, **kwargs)
    train_dataset = nf.ImageCoordinateDataset(path, **config["data"])
    warm_up(net, train_dataset, device)
    res = nf.train(
        net,
        train_dataset,
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
