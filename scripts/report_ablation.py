#!/usr/bin/env python
"""Report the FUTON ablations on the occupancy task.

Usage:
    python scripts/report_ablation.py
    python scripts/report_ablation.py --log-dir logs/ablation-futon

Writes results/ablation-futon/, one folder per study:
    basis/            the bases at the reference size (K=256, R=144)
    components_rank/  IoU against rank R at each K, and against K at each R
    tensor_net/       CP against tensor-ring combiners of equal size at K=256
Each folder holds a summary table and plots; bands and bars are one standard
error of the mean over the shapes.

Meshes of the ablation runs come from the occupancy report, e.g.
    python scripts/report_occupancy.py --log-dir logs/ablation-futon \
        --out-dir results/ablation-futon/basis --models FUTON-chebyshev
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns

import report_common as report

METRICS = ("iou",)
GRID = re.compile(r"FUTON-(?P<basis>\w+)-K(?P<components>\d+)-R(?P<rank>\d+)$")
RING = re.compile(r"FUTON-(?P<basis>\w+)-TR-K(?P<components>\d+)-R(?P<rank>\d+)$")
REFERENCE = re.compile(r"FUTON-(?P<basis>\w+)$")


def label(pattern: re.Pattern, data: pd.DataFrame) -> pd.DataFrame:
    """Keep the rows whose model matches, adding the pattern's fields."""
    fields = data["model"].str.extract(pattern)
    rows = data.join(fields.dropna()).dropna(subset=fields.columns.tolist())
    for column in ("components", "rank"):
        if column in rows:
            rows[column] = rows[column].astype(int)
    return rows


def grid_plots(data: pd.DataFrame, metric: str, out_dir: Path) -> None:
    """IoU against rank at each number of components, and the other way round."""
    for basis, rows in data.groupby("basis"):
        for x, hue in (("rank", "components"), ("components", "rank")):
            figure, plot = plt.subplots(figsize=(3.6, 3.0))
            sns.lineplot(
                data=rows,
                x=x,
                y=metric,
                hue=hue,
                style=hue,
                markers=True,
                dashes=False,
                errorbar="se",
                palette=list(report.RAMP[: rows[hue].nunique()]),
                markersize=5,
                markeredgecolor="white",
                markeredgewidth=0.5,
                ax=plot,
            )
            labels = {"rank": "Rank $R$", "components": "Components $K$"}
            plot.set(
                xscale="log",
                xlabel=labels[x],
                ylabel=report.METRICS[metric][0],
                title=f"FUTON-{basis}",
            )
            # The levels double, so tick them exactly, without log minor ticks.
            plot.set_xticks(sorted(rows[x].unique()))
            plot.set_xticklabels(sorted(rows[x].unique()))
            plot.xaxis.set_minor_locator(ticker.NullLocator())
            plot.legend(title=labels[hue].split()[-1], frameon=False)
            report.save(figure, out_dir / f"{metric}_vs_{x}_{basis}")


def size_plots(data: pd.DataFrame, metric: str, out_dir: Path) -> None:
    """IoU against model size for the CP and tensor-ring combiners, per basis."""
    data = data.assign(
        combiner=data["model"].str.contains("-TR-").map({True: "TR", False: "CP"}),
        basis=data["model"].str.extract(r"FUTON-(\w+)")[0],
    )
    for basis, rows in data.groupby("basis"):
        figure, plot = plt.subplots(figsize=(3.6, 3.0))
        sns.lineplot(
            data=rows,
            x="parameters",
            y=metric,
            hue="combiner",
            style="combiner",
            markers=True,
            dashes=False,
            errorbar="se",
            palette=list(report.PALETTE[:2]),
            markersize=5,
            markeredgecolor="white",
            markeredgewidth=0.5,
            ax=plot,
        )
        plot.set(
            xlabel="Parameters (k)",
            ylabel=report.METRICS[metric][0],
            title=f"FUTON-{basis}, K = 256",
        )
        # The sizes roughly double, so tick them on a log axis.
        sizes = sorted(rows["parameters"].unique())
        plot.set_xscale("log")
        plot.set_xticks(sizes, [f"{size:.0f}" for size in sizes])
        plot.xaxis.set_minor_locator(ticker.NullLocator())
        plot.legend(title="Combiner", frameon=False)
        report.save(figure, out_dir / f"{metric}_vs_size_{basis}")


def study(
    runs: list[dict], models: list[str], metrics, out_dir: Path, plots: bool = True
) -> pd.DataFrame:
    """Table and convergence plots for one group of models."""
    chosen = [run for run in runs if run["model"] in models]
    data = report.results(chosen)
    summary = report.table(data, metrics, out_dir, report.SPEED["occupancy"])
    if plots:
        report.convergence(
            report.curves(chosen), metrics, out_dir, list(summary.index), time_limit=15
        )
    return summary


def main() -> None:
    parser = report.parser("ablation-futon", "ablation-futon", "lucy")
    args = parser.parse_args()

    runs = report.read(args.log_dir, "occupancy", args.models)
    data = report.results(runs)
    report.style()
    metric = METRICS[0]

    # Rank and components: a grid, so curves against R and against K.
    grid = label(GRID, data)
    grid_plots(grid, metric, args.out_dir / "components_rank")
    report.table(
        grid, METRICS, args.out_dir / "components_rank", report.SPEED["occupancy"]
    )
    print(f"rank x components: {grid['model'].nunique()} models")

    # CP against tensor ring at K=256. A ring of rank r has the size of a CP
    # combiner of rank r^2; CP rank 144 is the reference model itself.
    rings = label(RING, data)
    partners = {
        f"FUTON-{basis}-K256-R{rank ** 2}" if rank**2 != 144 else f"FUTON-{basis}"
        for basis, rank in zip(rings["basis"], rings["rank"])
    }
    networks = sorted(set(rings["model"]) | (partners & set(data["model"])))
    study(runs, networks, METRICS, args.out_dir / "tensor_net", plots=False)
    for basis in sorted(rings["basis"].unique()):
        chosen = [m for m in networks if m.startswith(f"FUTON-{basis}")]
        curves = report.curves([run for run in runs if run["model"] in chosen])
        report.convergence(
            curves, METRICS, args.out_dir / "tensor_net" / basis, time_limit=15
        )
    size_plots(data[data["model"].isin(networks)], metric, args.out_dir / "tensor_net")
    print(f"tensor network: {len(networks)} models")

    # Bases at the reference size.
    bases = sorted(set(label(REFERENCE, data)["model"]))
    study(runs, bases, METRICS, args.out_dir / "basis")
    print(f"basis: {len(bases)} models")


if __name__ == "__main__":
    main()
