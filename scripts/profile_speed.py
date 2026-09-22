"""Time every model's inference on one signal per task, under one allocation.

The benchmark's own durations come from jobs that shared their node, and
rendering is partly bound by kernel launches, so they carry whatever else ran
beside them. Timing every model in turn inside one job makes the comparison
even; an exclusive allocation makes it clean:

    sbatch --clusters=accelgor --exclusive --time=1:00:00 --gpus-per-node=1 \\
      --chdir=$VSC_DATA/projects/neurofield --output=logs/slurm/%x-%j.out \\
      --job-name=nf-speed --wrap \\
      "source $VSC_DATA/venvs/neurofield-env/bin/activate && python scripts/profile_speed.py"

Usage:
    python scripts/profile_speed.py
    python scripts/profile_speed.py --tasks image occupancy --repeats 9

Writes logs/speed/<task>.json: each model's median inference time over the
repeats, the points it evaluated, and the rate those give.
"""

import argparse
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import report_paper as report
import torch
from report_paper import field_and_renderer

import neurofield as nf

CHUNK = 262_144  # coordinates per forward pass, as configs/occupancy.yaml uses


def timed(call: Callable[[], object], repeats: int) -> float:
    """Median seconds of a call, after two warm-ups, synchronized on CUDA."""
    durations = []
    for _ in range(repeats + 2):
        torch.cuda.synchronize()
        start = time.perf_counter()
        call()
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - start)
    return statistics.median(durations[2:])


def image_inference(run: dict, path: Path, device) -> tuple[Callable, int]:
    """Render the whole image from its coordinates, as the benchmark scores it."""
    dataset = nf.ImageCoordinateDataset(path)
    model = report.load_model(run, 2, 3, device)
    coords = dataset.input.to(device)
    return (
        lambda: nf.chunked_inference(model, coords, device=device),
        coords[..., 0].numel(),
    )


def occupancy_inference(run: dict, path: Path, device) -> tuple[Callable, int]:
    """Evaluate the whole volume in chunks, as the benchmark scores it."""
    config = report.model_config(run)
    dataset = nf.OccupancyCoordinateDataset(path, **config["data"] | {"subsample": 1.0})
    model = report.load_model(run, 3, 1, device)
    coords = dataset.input.to(device)
    return (
        lambda: nf.chunked_inference(model, coords, chunk_size=CHUNK, device=device),
        coords[..., 0].numel(),
    )


def nerf_inference(run: dict, path: Path, device) -> tuple[Callable, int]:
    """Render one test view, as the benchmark scores it.

    ``field_and_renderer`` rebuilds the occupancy grid from the checkpointed
    field, so the timing covers rendering alone.
    """
    config = report.model_config(run)
    dataset = nf.nerf.BlenderDataset(
        path, "test", downsample=config["data"]["downsample"]
    )
    field, renderer = field_and_renderer(run, config, dataset, device)
    batch = dataset[0]
    background = dataset.background.to(device)

    def render() -> object:
        with torch.no_grad(), nf.nerf.evaluation._eval_mode(field, renderer):
            return nf.nerf.render_image(
                field, renderer, batch["rays_o"], batch["rays_d"], background=background
            )

    return render, batch["rays_o"][..., 0].numel()


INFERENCE = {
    "image": image_inference,
    "occupancy": occupancy_inference,
    "nerf": nerf_inference,
}
UNITS = {"image": "pixels", "occupancy": "voxels", "nerf": "rays"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks", nargs="+", default=list(INFERENCE), choices=list(INFERENCE)
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--log-dir", type=Path, default=report.ROOT / "logs")
    parser.add_argument("--out-dir", type=Path, default=report.ROOT / "logs" / "speed")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        print(torch.cuda.get_device_name(device), flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for task in args.tasks:
        signal = report.EXAMPLES[task][0]
        path = report.ROOT / report.SIGNALS[task].format(signal)
        runs = [
            run
            for run in report.read(args.log_dir / task, task)
            if run["data"] == signal
        ]
        print(f"\n{task} ({signal}), median of {args.repeats}")
        measured = {}
        for run in sorted(runs, key=lambda run: run["model"]):
            call, points = INFERENCE[task](run, path, device)
            seconds = timed(call, args.repeats)
            measured[run["model"]] = {
                "seconds": seconds,
                "points": points,
                "rate": points / seconds,
            }
            print(
                f"  {run['model']:14} {1000 * seconds:8.1f} ms"
                f"  {points / seconds / 1e6:7.2f} M{UNITS[task]}/s",
                flush=True,
            )
            torch.cuda.empty_cache()
        (args.out_dir / f"{task}.json").write_text(
            json.dumps(measured, indent=2, sort_keys=True) + "\n"
        )
    print(f"\nWritten to {args.out_dir}")


if __name__ == "__main__":
    main()
