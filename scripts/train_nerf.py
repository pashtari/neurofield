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
from train_common import ROOT, build, fresh, record, run, warm_up_field

import neurofield as nf

DATA = sorted(
    p.parent for p in (ROOT / "data/nerf/blender").glob("*/transforms_train.json")
)


def fresh_targets(
    dataset: nf.nerf.BlenderDataset, device: torch.device, count: int = 10
) -> list[torch.Tensor]:
    """FreSh's targets for a radiance field: training views drawn at random,
    without repeats, composited onto black as the authors' code does."""
    views = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(0))
    black = torch.zeros(3)
    return [
        dataset.composite(dataset.images[i], black).to(device) for i in views[:count]
    ]


@torch.no_grad()
def fresh_view(
    field: nf.nerf.RadianceField, dataset: nf.nerf.BlenderDataset, device: torch.device
) -> torch.Tensor:
    """FreSh's render of an untrained field (Kania et al., ICLR 2025): a random
    training view whose rays are each sampled once, at the cameras' mean
    distance from the origin, and coloured without their view direction."""
    view = dataset[int(torch.randint(len(dataset), ()))]
    points = (view["rays_o"] + dataset.radius * view["rays_d"]).to(device)
    field = field.to(device)
    features = field.density_net(field.normalize_coordinates(points))[..., 1:]
    blank = features.new_zeros(
        *features.shape[:-1], field.direction_encoding.out_features
    )
    return torch.sigmoid(field.color_net(torch.cat([features, blank], dim=-1)))


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

    model_class, kwargs = build(model)
    search = fresh(
        model,
        kwargs,
        fresh_targets(train_dataset, device),
        lambda kw: fresh_view(
            nf.nerf.RadianceField(
                density_net=(model_class, kw),
                aabb=train_dataset.aabb,
                **config["field"],
            ),
            train_dataset,
            device,
        ),
        config["train"]["seed"],
    )
    field = nf.nerf.RadianceField(
        density_net=(model_class, kwargs), aabb=train_dataset.aabb, **config["field"]
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
    return {**record(res, model, test["mean"], search), "test_views": test["views"]}


if __name__ == "__main__":
    run("nerf", DATA, fit)
