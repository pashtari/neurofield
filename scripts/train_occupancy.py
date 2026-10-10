"""Occupancy benchmark: fit each model in configs/occupancy.yaml to Stanford shapes.

Usage:
    python scripts/train_occupancy.py
    python scripts/train_occupancy.py --data data/occupancy/lucy.ply --models SIREN --device cuda:1
"""

from pathlib import Path
from typing import Any

import torch
from train_common import ROOT, build, fresh, on_grid, record, run, warm_up

import neurofield as nf

DATA = sorted((ROOT / "data" / "occupancy").glob("*.ply"))


def fit(
    config: dict, model: dict, path: Path, out: Path, device: torch.device
) -> dict[str, Any]:
    model_class, kwargs = build(model)
    data_kwargs = {**config["data"], "item_id": path.stem}
    volume = nf.OccupancyCoordinateDataset(path, **{**data_kwargs, "subsample": 1.0})
    search = fresh(
        model,
        kwargs,
        [volume.target.to(device)],
        lambda kw: on_grid(model_class(3, 1, **kw), volume, device),
        config["train"]["seed"],
    )
    net = model_class(in_features=3, out_features=1, **kwargs)
    train_dataset = nf.OccupancyCoordinateDataset(path, **data_kwargs)
    warm_up(net, train_dataset, device)
    res = nf.train(
        net,
        train_dataset,
        volume,
        metrics={"iou": nf.iou},
        # Rendering a mesh needs a display, which compute nodes lack; recreate
        # reconstructions from checkpoint.pt instead.
        save_reconstruction=False,
        device=device,
        log_dir=out,
        **{**config["train"], **model["train"]},
    )
    return record(res, model, res["history"][-1]["eval"], search)


if __name__ == "__main__":
    run("occupancy", DATA, fit)
