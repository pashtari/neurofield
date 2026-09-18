#!/usr/bin/env python
"""Report the image benchmark: table, convergence plots, and example images.

Usage:
    python scripts/report_image.py
    python scripts/report_image.py --qualitative kodim17 kodim04

Writes results/image/: summary.{csv,md,tex} averaged over the images,
<metric>_{iteration,time},
and qualitative/<image>/ with the original and each model's reconstruction,
named by rank, model and PSNR.
"""

from pathlib import Path

import neurofield as nf
import report_common as report

METRICS = ("psnr", "ssim", "ms_ssim", "lpips")


def qualitative(runs: list[dict], signals: list[str], out_dir: Path, device) -> None:
    """Save the original image and every model's reconstruction."""
    for signal, ranked in report.examples(runs, signals, METRICS[0]).items():
        directory = out_dir / "qualitative" / signal
        directory.mkdir(parents=True, exist_ok=True)
        dataset = nf.ImageCoordinateDataset(
            report.ROOT / report.SIGNALS["image"].format(signal)
        )
        dataset.save(dataset.original, directory / "00_ground_truth")
        panels = [("Ground truth", directory / "00_ground_truth.png")]

        print(f"{signal}: rendering {len(ranked)} reconstructions", flush=True)
        for rank, run in enumerate(ranked, start=1):
            print(f"  {run['model']}", flush=True)
            model = report.load_model(run, 2, 3, device)
            output = nf.chunked_inference(model, dataset.input, device=device)
            name = report.figure_name(
                rank, run["model"], "psnr", run["metrics"]["psnr"]
            )
            dataset.save(dataset.postprocess(output.cpu()), directory / name)
            panels.append(
                (
                    f"{run['model']}\n{run['metrics']['psnr']:.2f} dB",
                    directory / f"{name}.png",
                )
            )
        report.montage(panels, directory / "comparison")
        print(f"{signal}: written to {directory}")


def main() -> None:
    args = report.parser("image", "image", "kodim17").parse_args()
    runs = report.read(args.log_dir, "image", args.models)
    summary = report.report(
        runs, METRICS, args.out_dir, "image", per_signal=False, time_limit=15
    )
    print(summary.to_string())
    if args.qualitative != ["none"]:
        qualitative(runs, args.qualitative, args.out_dir, args.device)


if __name__ == "__main__":
    main()
