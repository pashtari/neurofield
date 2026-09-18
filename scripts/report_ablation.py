#!/usr/bin/env python
"""Report the FUTON ablations on the occupancy task.

Usage:
    python scripts/report_ablation.py
    python scripts/report_ablation.py --log-dir logs/ablation-futon

Writes results/ablation-futon/, one folder per study:
    components_rank/  IoU against rank R at each K, and against K at each R
    tensor_net/       the tensor-ring against the CP combiners of equal size
    basis/            the bases at the reference size
Each folder holds summary tables and, except for the grid, convergence plots.
Bands and bars are the standard error of the mean over the shapes.

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
            figure, plot = plt.subplots(figsize=(6, 4.5))
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
                markersize=8,
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
            plot.legend(title=labels[hue].split()[-1], fontsize=9)
            report.save(figure, out_dir / f"{metric}_vs_{x}_{basis}")


def study(
    runs: list[dict], models: list[str], metrics, out_dir: Path, plots: bool = True
) -> pd.DataFrame:
    """Table and convergence plots for one group of models."""
    chosen = [run for run in runs if run["model"] in models]
    data = report.results(chosen)
    summary = report.table(data, metrics, out_dir, report.SPEED["occupancy"])
    if plots:
        report.convergence(
            report.curves(chosen), metrics, out_dir, list(summary.index), time_limit=10
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

    # Tensor ring against the CP combiners of the same size.
    rings = label(RING, data)
    pairs = sorted(
        set(rings["model"])
        | {f"FUTON-{basis}" for basis in rings["basis"]}  # CP R=144, matching TR r=12
        | {
            f"FUTON-{row.basis}-K{row.components}-R{row.rank ** 2}"
            for row in rings.itertuples()
        }
    )
    study(
        runs,
        [name for name in pairs if name in set(data["model"])],
        METRICS,
        args.out_dir / "tensor_net",
    )
    print(f"tensor ring: {len(pairs)} models")

    # Bases at the reference size.
    bases = sorted(set(label(REFERENCE, data)["model"]))
    study(runs, bases, METRICS, args.out_dir / "basis")
    print(f"basis: {len(bases)} models")


if __name__ == "__main__":
    main()
