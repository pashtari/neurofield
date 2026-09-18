#!/usr/bin/env python
"""Report the occupancy benchmark: table, convergence plots, and example shapes.

Usage:
    python scripts/report_occupancy.py
    python scripts/report_occupancy.py --qualitative lucy thai_statue

Writes results/occupancy/: summary.{csv,md,tex}, iou_{iteration,time},
and qualitative/<shape>/ with the original and each model's mesh, named by
rank, model and IoU. Rendering a mesh to PNG needs a display, so run this on a
workstation; on a compute node only the .dae mesh is written.
"""

from pathlib import Path

import neurofield as nf
import report_common as report

METRICS = ("iou",)


def qualitative(runs: list[dict], signals: list[str], out_dir: Path, device) -> None:
    """Save the original shape and every model's reconstruction as mesh and render."""
    for signal, ranked in report.examples(runs, signals, "iou").items():
        directory = out_dir / "qualitative" / signal
        directory.mkdir(parents=True, exist_ok=True)
        resolution = report.model_config(ranked[0])["data"]["resolution"]
        dataset = nf.OccupancyCoordinateDataset(
            report.ROOT / report.SIGNALS["occupancy"].format(signal),
            resolution=resolution,
            item_id=signal,
        )
        dataset.save(dataset.original, directory / "00_ground_truth")
        panels = [("Ground truth", directory / "00_ground_truth.png")]

        print(f"{signal}: rendering {len(ranked)} meshes", flush=True)
        for rank, run in enumerate(ranked, start=1):
            print(f"  {run['model']}", flush=True)
            model = report.load_model(run, 3, 1, device)
            output = nf.chunked_inference(
                model, dataset.input, chunk_size=512**2, device=device
            )
            value = report.scale("iou", run["metrics"]["iou"])
            name = report.figure_name(rank, run["model"], "iou", value)
            dataset.save(dataset.postprocess(output.cpu()), directory / name)
            panels.append(
                (f"{run['model']}\n{value:.2f}% IoU", directory / f"{name}.png")
            )
        if all(image.exists() for _, image in panels):  # rendering needs a display
            report.montage(panels, directory / "comparison")
        print(f"{signal}: written to {directory}")


def main() -> None:
    args = report.parser("occupancy", "occupancy", "lucy").parse_args()
    runs = report.read(args.log_dir, "occupancy", args.models)
    summary = report.report(runs, METRICS, args.out_dir, "occupancy", time_limit=15)
    print(summary.to_string())
    if args.qualitative != ["none"]:
        qualitative(runs, args.qualitative, args.out_dir, args.device)


if __name__ == "__main__":
    main()
