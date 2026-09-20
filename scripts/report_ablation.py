"""Report the FUTON ablations on the occupancy task.

Usage:
    python scripts/report_ablation.py

Each study has a config, configs/ablation-futon/<study>.yaml, runs in
logs/ablation-futon/<study>/, and results in results/ablation-futon/<study>/:
    basis/            every basis at the benchmark's size (K=128, R=218)
    components_rank/  IoU against rank R at each K, and against K at each R
    tensor_net/       the default CP combiner against a tensor ring, per basis
Each folder holds a summary table and plots; bands and bars are one standard
error of the mean over the shapes. Only runs whose setup matches the study's
config count, so runs of since-changed entries drop out, and studies without
runs are skipped.

Meshes of the ablation runs come from the occupancy report, e.g.
    python scripts/report_occupancy.py --log-dir logs/ablation-futon/basis \
        --out-dir results/ablation-futon/basis --models FUTON-chebyshev
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import report_common as report
import seaborn as sns
import yaml
from matplotlib import ticker

METRIC = "iou"
STUDIES = ("basis", "components_rank", "tensor_net")
GRID = re.compile(r"FUTON-(?P<basis>\w+)-K(?P<components>\d+)-R(?P<rank>\d+)$")
RING = re.compile(r"FUTON-(?P<basis>\w+)-TR$")


def read(study: str, log_dir: Path, models: list[str] | None) -> list[dict]:
    """The study's runs whose setup matches its config."""
    config = report.ROOT / "configs" / "ablation-futon" / f"{study}.yaml"
    setups = yaml.safe_load(config.read_text())["models"]
    if not (log_dir / study).exists():
        return []
    return [
        run
        for run in report.read(log_dir / study, "occupancy")
        if setups.get(run["model"]) == run["setup"]
        and (not models or run["model"] in models)
    ]


def table(runs: list[dict], out_dir: Path) -> pd.DataFrame:
    return report.table(
        report.results(runs), (METRIC,), out_dir, report.SPEED["occupancy"]
    )


def convergence(runs: list[dict], out_dir: Path, models=None) -> None:
    report.convergence(report.curves(runs), (METRIC,), out_dir, models, time_limit=15)


def grid_plots(runs: list[dict], out_dir: Path) -> None:
    """IoU against rank at each number of components, and the other way round."""
    data = report.results(runs)
    data = data.join(data["model"].str.extract(GRID))
    data[["components", "rank"]] = data[["components", "rank"]].astype(int)
    labels = {"rank": "Rank $R$", "components": "Components $K$"}
    for basis, rows in data.groupby("basis"):
        for x, hue in (("rank", "components"), ("components", "rank")):
            figure, plot = plt.subplots(figsize=(3.6, 3.0))
            sns.lineplot(
                data=rows,
                x=x,
                y=METRIC,
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
            plot.set(
                xscale="log",
                xlabel=labels[x],
                ylabel=report.METRICS[METRIC][0],
                title=f"FUTON-{basis}",
            )
            # The levels double, so tick them exactly, without log minor ticks.
            plot.set_xticks(sorted(rows[x].unique()))
            plot.set_xticklabels(sorted(rows[x].unique()))
            plot.xaxis.set_minor_locator(ticker.NullLocator())
            plot.legend(title=labels[hue].split()[-1], frameon=False)
            report.save(figure, out_dir / f"{METRIC}_vs_{x}_{basis}")


def main() -> None:
    args = report.parser(
        "ablation-futon", "ablation-futon", report.EXAMPLE["occupancy"]
    ).parse_args()
    report.style()
    for study in STUDIES:
        runs = read(study, args.log_dir, args.models)
        out_dir = args.out_dir / study
        print(f"{study}: {len({run['model'] for run in runs})} models")
        if not runs:
            continue
        summary = table(runs, out_dir)
        if study == "basis":
            convergence(runs, out_dir, list(summary.index))
        elif study == "components_rank":
            grid_plots(runs, out_dir)
        else:  # CP, the default, against TR for each basis
            bases = {
                match["basis"] for run in runs if (match := RING.match(run["model"]))
            }
            for basis in sorted(bases):
                pair = (f"FUTON-{basis}", f"FUTON-{basis}-TR")
                convergence(
                    [run for run in runs if run["model"] in pair], out_dir / basis
                )


if __name__ == "__main__":
    main()
