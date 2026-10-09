"""FUTON's error bound against trained FUTONs, on Kodak images and occupancy volumes.

Usage:
    python scripts/futon_bound.py                  # every signal and check
    python scripts/futon_bound.py --tasks image --signals kodim19 kodim21
    python scripts/report_paper.py --tasks bound   # then the figures and tables

For every signal, spectral resolution K = (alpha N_1, ..., alpha N_C) and CP
rank R, the script bounds the best grid error of the paper's FUTON, a sinc
basis, a CP combiner and a linear decoder without bias (``nf.futon_bound``);
evaluates the certified FUTON, which attains the upper bound and is built
from the bound's factors without training, in its ALS form, the default, and
times its SVD form, the same terms before alternating least squares refines
them; and trains the same FUTON with Adam from a random start, as the
benchmark trains every model. A check completes the study: on
every signal at one setting, longer training and alternating least squares
on the whole grid, to see whether training rather than the bound is the
limit. Errors are mean squared errors on the whole grid, and PSNR is taken
for signals in [-1, 1]. The theorem behind the bound is in
docs/concepts/error-bound.md, and scripts/report_paper.py turns the records
into figures and tables that report each dataset, its signals averaged.

Writes:
    logs/bound/<task>/<signal>.json  one record per (alpha, R), resumable
    logs/bound/<task>_checks.json  the optimizer checks, one row per signal
"""

import argparse
import json
import os
import time

# nf.train's progress bar, off when the study runs, but not where it is imported
if __name__ == "__main__":
    os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import torch

import neurofield as nf
from train_common import ROOT

TASKS = {
    "image": {
        "signals": [f"kodim{i:02d}" for i in range(1, 25)],
        "alphas": (0.125, 0.25, 0.5),
        "ranks": (16, 32, 64, 128, 256),
        "subsample": 0.1,
    },
    "occupancy": {
        "signals": ["armadillo", "dragon", "happy_buddha", "lucy", "thai_statue"],
        "alphas": (0.125, 0.25, 0.5),
        "ranks": (32, 64, 128, 256, 512),
        "subsample": 0.01,
    },
}
EPOCHS, LR = 2000, 3e-2  # the benchmark's protocol; 3e-2 is best of the grid here
# The optimizer checks at K = N/2 and the middle rank of each task: Adam for
# LONGER times the epochs, and alternating least squares on the whole grid,
# read after each number of SWEEPS.
CHECKS = {"image": 64, "occupancy": 128}
LONGER, SWEEPS = 5, (500, 2000)
LOGS = ROOT / "logs" / "bound"


def psnr(mse: float | np.ndarray) -> float | np.ndarray:
    """PSNR of a mean squared error, for signals in [-1, 1]."""
    return 10 * np.log10(4 / np.asarray(mse))


def dataset(task: str, name: str, subsample: float = 1.0):
    if task == "image":
        return nf.ImageCoordinateDataset(
            ROOT / f"data/Kodak/{name}.png", subsample=subsample
        )
    path = ROOT / f"data/occupancy/{name}.ply"
    return nf.OccupancyCoordinateDataset(path, resolution=256, subsample=subsample)


def futon(sizes: list[int], channels: int, alpha: float, rank: int) -> nf.FUTON:
    """The paper's FUTON with the benchmark's sinc basis: a CP combiner and a
    linear decoder, no bias."""
    components = [max(2, int(alpha * size)) for size in sizes]
    basis = ("sinc", {"num_components": components, "grid_size": sizes})
    return nf.FUTON(len(sizes), channels, basis, ("cp", {"rank": rank}),
                    ("linear", {"bias": False}))  # fmt: skip


def features_of(model: nf.FUTON, sizes: list[int], device) -> list[torch.Tensor]:
    """The model's basis on each axis's grid, as ``nf.futon_bound`` takes it."""
    return [
        model.basis.axis_features(torch.linspace(-1, 1, n), c).to(device)
        for c, n in enumerate(sizes)
    ]


def evaluate(model: nf.FUTON, data, device) -> tuple[float, float | None]:
    """Grid MSE, and IoU (%) at the zero level for occupancy."""
    output = nf.chunked_inference(model, data.input, chunk_size=2**20, device=device)
    target = data.target.to(device)
    mse = float((output.double() - target.double()).square().mean())
    if target.shape[-1] != 1 or not torch.all(target.abs() == 1):
        return mse, None
    inside, truth = output > 0, target > 0
    return mse, 100 * float((inside & truth).sum() / (inside | truth).sum())


def train(model: nf.FUTON, data, device, lr: float = LR, epochs: int = EPOCHS,
          seed: int = 0) -> float:  # fmt: skip
    """Train in place with Adam as the benchmark does; returns the seconds taken."""
    torch.manual_seed(seed)
    model.combiner.reset_parameters()
    model.decoder.reset_parameters()
    start = time.perf_counter()
    nf.train(model, data, num_epochs=epochs, lr=lr, log_interval=0,
             save_reconstruction=False, device=device, seed=seed)  # fmt: skip
    return time.perf_counter() - start


def sinc_only(configs: list[dict]) -> list[dict]:
    """The records' sinc settings; an earlier study also ran a lanczos basis."""
    return [c for c in configs if c.get("basis", "sinc") == "sinc"]


def run(task: str, name: str, device: torch.device) -> None:
    """Bound, certify and train every setting of one signal not yet recorded."""
    settings = TASKS[task]
    path = LOGS / task / f"{name}.json"
    record = json.loads(path.read_text()) if path.exists() else {"configs": []}
    done = {(c["alpha"], c["R"]) for c in sinc_only(record["configs"])}
    todo = [
        (alpha, rank)
        for alpha in settings["alphas"]
        for rank in settings["ranks"]
        if (alpha, rank) not in done
    ]
    if not todo:
        return
    full = dataset(task, name)
    train_data = dataset(task, name, settings["subsample"])
    sizes, channels = list(full.target.shape[:-1]), full.target.shape[-1]
    record.update(task=task, signal=name, sizes=sizes, channels=channels)
    target = full.target.to(device)
    for alpha in sorted({alpha for alpha, _ in todo}):
        ranks = [rank for a, rank in todo if a == alpha]
        features = features_of(futon(sizes, channels, alpha, 1), sizes, device)
        start = time.perf_counter()
        bound = nf.futon_bound(target, features, ranks)
        seconds = time.perf_counter() - start
        start = time.perf_counter()  # the SVD form, before any refinement
        closed = nf.futon_bound(target, features, ranks, sweeps=0)
        closed_seconds = time.perf_counter() - start
        for i, (rank, lower, upper) in enumerate(
            zip(bound.ranks, bound.lower, bound.upper)
        ):
            certified = futon(sizes, channels, alpha, rank)
            bound.initialize(certified)
            certified_mse, certified_iou = evaluate(certified, full, device)
            model = futon(sizes, channels, alpha, rank)
            adam_seconds = train(model, train_data, device)
            adam_mse, adam_iou = evaluate(model, full, device)
            config = {
                "alpha": alpha,
                "K": [max(2, int(alpha * n)) for n in sizes],
                "R": rank,
                "parameters": nf.count_parameters(model),
                "truncation": bound.truncation,
                "lower": float(lower),
                "upper": float(upper),
                "bound_seconds": seconds,
                "certified": certified_mse,
                "certified_iou": certified_iou,
                "adam": adam_mse,
                "adam_iou": adam_iou,
                "adam_seconds": adam_seconds,
                "closed": float(closed.upper[i]),  # the SVD form's error
                "closed_seconds": closed_seconds,
            }
            record["configs"].append(config)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, indent=1))
            print(f"{task} {name} alpha {alpha} R {rank}: certified "
                  f"[{psnr(upper):.2f}, {psnr(lower):.2f}] dB, Adam "
                  f"{psnr(adam_mse):.2f} dB", flush=True)  # fmt: skip


def checks(task: str, device: torch.device) -> None:
    """Whether training, not the bound, is the limit, on every signal at
    K = N/2 and the task's CHECKS rank: Adam for LONGER times the benchmark's
    epochs, and alternating least squares on the whole grid from a random
    start (``nf.LeastSquares`` without momentum), against the bound."""
    path = LOGS / f"{task}_checks.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    rank = CHECKS[task]
    for name in TASKS[task]["signals"]:
        if any(row["signal"] == name for row in rows):
            continue
        full = dataset(task, name)
        sizes, channels = list(full.target.shape[:-1]), full.target.shape[-1]
        batch = {"input": full.input.to(device), "target": full.target.to(device)}
        model = futon(sizes, channels, 0.5, rank)
        bound = nf.futon_bound(batch["target"], features_of(model, sizes, device), rank)
        row = {"signal": name, "R": rank, "upper": float(bound.upper[0]),
               "lower": float(bound.lower[0])}  # fmt: skip
        train_data = dataset(task, name, TASKS[task]["subsample"])
        train(model, train_data, device, epochs=LONGER * EPOCHS)
        row["adam_longer"] = evaluate(model, full, device)[0]
        torch.manual_seed(0)
        solver = nf.LeastSquares(model.to(device), momentum=0.0)
        for sweep in range(1, max(SWEEPS) + 1):
            solver.step(batch)
            if sweep in SWEEPS:
                row[f"als_{sweep}"] = evaluate(model, full, device)[0]
        rows.append(row)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, indent=1))
        print(f"{task} checks {name} done", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    parser.add_argument("--signals", nargs="+", help="signals to run (default: all)")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    for task in args.tasks:
        for name in args.signals or TASKS[task]["signals"]:
            if name in TASKS[task]["signals"]:
                run(task, name, device)
        if not args.signals:
            checks(task, device)


if __name__ == "__main__":
    main()
