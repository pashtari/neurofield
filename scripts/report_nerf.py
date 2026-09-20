"""Report the NeRF benchmark: table, convergence plots, and example renders.

Usage:
    python scripts/report_nerf.py
    python scripts/report_nerf.py --qualitative lego --frames 60

Writes results/nerf/: summary.{csv,md,tex}, <metric>_{iteration,time},
and qualitative/<scene>/ with a ground-truth test view and, per model, that
view and an orbit GIF, named by rank, model and PSNR. Rendering needs no
display, so this also runs on a compute node.
"""

from pathlib import Path

import numpy as np
import report_common as report
import torch
from PIL import Image

import neurofield as nf

METRICS = ("psnr", "ssim", "lpips")


def field_and_renderer(run: dict, config: dict, dataset, device):
    """Rebuild a run's radiance field and its renderer."""
    field = nf.nerf.RadianceField(
        density_net=report.build(run["setup"]), aabb=dataset.aabb, **config["field"]
    )
    state = torch.load(run["directory"] / "checkpoint.pt", map_location="cpu")
    field.load_state_dict(state)
    renderer = nf.nerf.create_renderer(
        backend="nerfacc" if nf.nerf.is_nerfacc_available() else "pytorch",
        aabb=dataset.aabb,
        near=dataset.near,
        far=dataset.far,
    )
    return field.to(device), renderer.to(device)


def qualitative(
    runs: list[dict], signals: list[str], out_dir: Path, device, frames: int
) -> None:
    """Save a ground-truth view, each model's render of it, and an orbit GIF."""
    for signal, ranked in report.examples(runs, signals, "psnr").items():
        directory = out_dir / "qualitative" / signal
        directory.mkdir(parents=True, exist_ok=True)
        config = report.model_config(ranked[0])
        dataset = nf.nerf.BlenderDataset(
            report.ROOT / report.SIGNALS["nerf"].format(signal),
            "test",
            downsample=config["data"]["downsample"],
        )
        view = config["render_views"][0]
        batch = dataset[view]
        target = (dataset.composite(batch["rgba"]) * 255).round().byte().numpy()
        Image.fromarray(target).save(directory / f"00_ground_truth_view{view:03d}.png")
        panels = [("Ground truth", directory / f"00_ground_truth_view{view:03d}.png")]

        # The dataset's own camera elevation and radius, in its scaled frame.
        poses = [
            nf.nerf.pose_spherical(theta, -30.0, dataset.radius)
            for theta in np.linspace(-180, 180, frames, endpoint=False)
        ]
        print(
            f"{signal}: rendering {len(ranked)} views"
            + (f" and {frames}-frame orbits" if frames else ""),
            flush=True,
        )
        for rank, run in enumerate(ranked, start=1):
            print(f"  {run['model']}", flush=True)
            field, renderer = field_and_renderer(run, config, dataset, device)
            name = report.figure_name(
                rank, run["model"], "psnr", run["metrics"]["psnr"]
            )

            rendered = nf.nerf.render_image(
                field,
                renderer,
                batch["rays_o"].to(device),
                batch["rays_d"].to(device),
                background=dataset.background.to(device),
            )
            image = (rendered["rgb"].clamp(0, 1) * 255).round().byte().cpu().numpy()
            Image.fromarray(image).save(directory / f"{name}_view{view:03d}.png")
            panels.append(
                (
                    f"{run['model']}\n{run['metrics']['psnr']:.2f} dB",
                    directory / f"{name}_view{view:03d}.png",
                )
            )

            if not frames:  # an orbit is a minute of rendering per model
                continue
            orbit = [
                Image.fromarray(frame)
                for frame in nf.nerf.render_views(field, renderer, dataset, poses)
            ]
            orbit[0].save(
                directory / f"{name}_orbit.gif",
                save_all=True,
                append_images=orbit[1:],
                duration=1000 // 15,
                loop=0,
            )
        report.montage(panels, directory / "comparison")
        print(f"{signal}: written to {directory}")


def main() -> None:
    parser = report.parser("nerf", "nerf", report.EXAMPLE["nerf"])
    parser.add_argument(
        "--frames", type=int, default=60, help="orbit frames; 0 skips the orbits"
    )
    args = parser.parse_args()

    runs = report.read(args.log_dir, "nerf", args.models)
    summary = report.report(runs, METRICS, args.out_dir, "nerf", time_limit=300)
    print(summary.to_string())
    if args.qualitative != ["none"]:
        qualitative(runs, args.qualitative, args.out_dir, args.device, args.frames)


if __name__ == "__main__":
    main()
