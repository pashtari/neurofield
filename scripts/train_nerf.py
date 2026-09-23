"""NeRF benchmark: fit each model in configs/nerf.yaml to Blender scenes.

Models are trained with validation views and scored on all test views.

Usage:
    python scripts/train_nerf.py
    python scripts/train_nerf.py --data data/nerf/blender/lego --models MFN --device cuda:1
"""

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from train_common import ROOT, build, record, run, warm_up_field

import neurofield as nf

DATA = sorted(
    p.parent for p in (ROOT / "data/nerf/blender").glob("*/transforms_train.json")
)


def fit(
    config: dict, model: dict, path: Path, out: Path, device: torch.device
) -> dict[str, Any]:
    splits = config["data"]
    downsample = splits["downsample"]
    train_dataset = nf.nerf.BlenderDataset(
        path, "train", downsample=downsample, skip=splits["train_skip"]
    )
    val_dataset = nf.nerf.BlenderDataset(path, "val", downsample=downsample)
    test_dataset = nf.nerf.BlenderDataset(path, "test", downsample=downsample)

    field = nf.nerf.RadianceField(
        density_net=build(model), aabb=train_dataset.aabb, **config["field"]
    )
    renderer = nf.nerf.create_renderer(
        backend="nerfacc" if nf.nerf.is_nerfacc_available() else "pytorch",
        aabb=train_dataset.aabb,
        near=train_dataset.near,
        far=train_dataset.far,
    )
    warm_up_field(field, train_dataset, device)
    res = nf.nerf.train(
        field,
        renderer,
        train_dataset,
        eval_dataset=val_dataset,
        eval_indices=list(range(0, len(val_dataset), splits["val_skip"])),
        device=device,
        log_dir=out,
        **{**config["train"], **model["train"]},
    )

    test = nf.nerf.evaluate(field, renderer, test_dataset, return_images=True)
    for view in config["render_views"]:
        image = (test["images"][view] * 255).round().byte().numpy()
        Image.fromarray(image).save(out / f"test_{view:03d}.png")
    return {**record(res, model, test["mean"]), "test_views": test["views"]}


if __name__ == "__main__":
    run("nerf", DATA, fit)
