"""Reading runs, and the tables, plots and figures shared by the reports.

The ``report_*.py`` scripts turn the run directories written by ``train_*.py``
into ``results/<report>/``: a table of the final metrics, convergence plots
averaged over signals, and qualitative examples rendered from the checkpoints.
"""

import argparse
import json
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pgf import LatexError
import pandas as pd
import seaborn as sns
import torch
import yaml

from train_common import ROOT, build, merge

# Header, whether larger is better, the factor the stored value is shown in,
# and the decimals of a table entry.
METRICS = {
    "psnr": ("PSNR (dB)", True, 1, 2),
    "ssim": ("SSIM (%)", True, 100, 2),
    "ms_ssim": ("MS-SSIM (%)", True, 100, 2),
    "lpips": ("LPIPS", False, 1, 4),
    "iou": ("IoU (%)", True, 100, 2),
}
SPEED = {"image": "img/s", "occupancy": "vol/s", "nerf": "views/s"}
# Categorical hues in a fixed order, and a light-to-dark blue ramp for ordered
# levels. Beyond eight models the hues repeat with a second line style, which
# keeps every series identifiable without inventing hues.
PALETTE = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
)
DASHES = ("", (4, 2), (1, 1.5))  # solid, dashed, dotted
RAMP = ("#9ec5f4", "#5598e7", "#2a78d6", "#184f95", "#0d366b")

SIGNALS = {  # where a run's signal lives, by task
    "image": "data/Kodak/{}.png",
    "occupancy": "data/occupancy/{}.ply",
    "nerf": "data/nerf/blender/{}",
}


def scale(metric: str, value: float) -> float:
    """A stored value as it is reported, e.g. IoU and SSIM as percentages."""
    return value * METRICS[metric][2]


def header(metric: str) -> str:
    """``PSNR (dB) up``: the name, its unit, and which direction is better."""
    name, larger_is_better, _, _ = METRICS[metric]
    arrow = "\u2191" if larger_is_better else "\u2193"
    return f"{name} {arrow}"


def entry(metric: str, mean: float, deviation: float | None = None) -> str:
    """One table cell, ``mean`` or ``mean+-std`` at the metric's precision."""
    decimals = METRICS[metric][3]
    if deviation is None or pd.isna(deviation):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f}\u00b1{deviation:.{decimals}f}"


def parser(report: str, default_log: str, signal: str) -> argparse.ArgumentParser:
    """Command line shared by the reports."""
    parser = argparse.ArgumentParser(description=f"Report the {report} runs.")
    parser.add_argument("--log-dir", type=Path, default=ROOT / "logs" / default_log)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / report)
    parser.add_argument("--models", nargs="+", help="models to report (default: all)")
    parser.add_argument(
        "--qualitative",
        nargs="+",
        default=[signal],
        metavar="SIGNAL",
        help=f"signals to render as examples (default: {signal}); none skips them",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser


def read(
    log_dir: Path, task: str, models: Sequence[str] | None = None
) -> list[dict[str, Any]]:
    """Load the ``task`` runs under ``log_dir``, each tagged with its directory."""
    runs = []
    for path in sorted(log_dir.rglob("results.json")):
        run = json.loads(path.read_text())
        if run["task"] != task or (models and run["model"] not in models):
            continue
        runs.append(run | {"directory": path.parent})
    if not runs:
        raise SystemExit(f"no {task} runs under {log_dir}")
    return runs


def results(runs: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per run: its signal, model, size, time and final metrics."""
    rows = []
    for run in runs:
        row = {
            "signal": run["data"],
            "model": run["model"],
            "parameters": run["num_params"] / 1000,
            "time": run["train_time"],
            # Evaluating one signal, so its reciprocal is the inference rate.
            "speed": 1 / run["metrics"]["duration"],
        }
        row |= {
            metric: scale(metric, value)
            for metric, value in run["metrics"].items()
            if metric in METRICS
        }
        rows.append(row)
    return pd.DataFrame(rows)


def curves(runs: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per evaluation: the metrics at that iteration and elapsed time."""
    rows = []
    for run in runs:
        for entry in run["history"]:
            if "eval" not in entry:
                continue
            row = {
                "signal": run["data"],
                "model": run["model"],
                "iteration": entry.get("epoch", entry.get("step")),
                "time": entry["elapsed"],
            }
            row |= {
                metric: scale(metric, value)
                for metric, value in entry["eval"].items()
                if metric in METRICS
            }
            rows.append(row)
    return pd.DataFrame(rows)


def order(table: pd.DataFrame, metric: str) -> list[str]:
    """Model names, best first, by the mean of ``metric`` over signals."""
    means = table.groupby("model")[metric].mean()
    return list(means.sort_values(ascending=not METRICS[metric][1]).index)


def save(figure: plt.Figure, path: Path) -> None:
    """Write a figure as PDF, and as PGF to include in a LaTeX document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    try:
        figure.savefig(path.with_suffix(".pgf"), bbox_inches="tight")
    except (RuntimeError, FileNotFoundError, LatexError) as error:  # PGF runs LaTeX
        warnings.warn(f"no {path.name}.pgf: {error}")
    plt.close(figure)


def markdown(summary: pd.DataFrame) -> str:
    """A Markdown table; its flat header joins the levels of a merged one."""
    columns = [
        (
            " ".join(part for part in column if part)
            if isinstance(column, tuple)
            else str(column)
        )
        for column in summary.columns
    ]
    lines = [[summary.index.name or "Model", *columns], ["---"] * (len(columns) + 1)]
    lines += [
        [str(index), *map(str, row)]
        for index, row in zip(summary.index, summary.to_numpy())
    ]
    return "".join("| " + " | ".join(line) + " |\n" for line in lines)


def table(
    data: pd.DataFrame,
    metrics: Sequence[str],
    out_dir: Path,
    speed: str = "img/s",
    per_signal: bool = True,
) -> pd.DataFrame:
    """One table of every model: size, time, speed, and each metric.

    The unlabelled first columns hold the size, the times, and each metric
    averaged over the signals as ``mean+-std``; after them one super column
    per signal holds that signal's metrics. CSV and LaTeX keep those two
    header levels; Markdown, which has no multi-column header, joins them.
    """
    models = order(data, metrics[0])
    grouped = data.groupby("model")
    mean = grouped.mean(numeric_only=True).loc[models]
    std = grouped.std(numeric_only=True).loc[models]

    columns = {
        ("", "# Params (k) \u2193"): mean["parameters"].map("{:.1f}".format),
        ("", "Time (s) \u2193"): mean["time"].map("{:.1f}".format),
        ("", f"Speed ({speed}) \u2191"): mean["speed"].map("{:.1f}".format),
    }
    for metric in metrics:  # averaged over the signals, so unlabelled
        columns[("", header(metric))] = pd.Series(
            {m: entry(metric, mean[metric][m], std[metric][m]) for m in models}
        )
    for signal, rows in data.groupby("signal") if per_signal else ():
        values = rows.set_index("model")
        for metric in metrics:
            columns[(signal, header(metric))] = values[metric].map(
                lambda value, metric=metric: entry(metric, value)
            )
    summary = pd.DataFrame(columns).loc[models].fillna("")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "summary.csv", index_label="Model")
    (out_dir / "summary.md").write_text(markdown(summary))
    # LaTeX reads #, % and _ as markup; none of the commands pandas emits
    # contains them, so escaping the whole table is safe.
    latex = summary.to_latex(multicolumn=True, multicolumn_format="c")
    for character in ("#", "%", "_"):
        latex = latex.replace(character, f"\\{character}")
    # Arrows and the deviation sign as commands, so pdflatex needs no extras.
    for character, command in (
        ("\u2191", r"$\uparrow$"),
        ("\u2193", r"$\downarrow$"),
        ("\u00b1", r"$\pm$"),
    ):
        latex = latex.replace(character, command)
    (out_dir / "summary.tex").write_text(latex)
    return summary


def y_range(
    means: pd.DataFrame, axis: str, larger_is_better: bool
) -> tuple[float, float]:
    """A range on which the models separate, from their mean curves.

    Early transients and a diverged model would each squeeze the rest together,
    so the weak end is trimmed to the tenth percentile of the second half of
    the run, then widened to every final value but a collapsed one.
    """
    converged = means[means[axis] >= means[axis].median()]["mean"]
    finals = means.groupby("model")["mean"].last()
    if larger_is_better:
        finals = finals[finals >= 0.5 * finals.max()]
        low, high = min(converged.quantile(0.10), finals.min()), means["mean"].max()
    else:
        finals = finals[finals <= 5 * finals.min()]
        low, high = means["mean"].min(), max(converged.quantile(0.90), finals.max())
    margin = 0.05 * (high - low)
    return low - margin, high + margin


def within_signal(data: pd.DataFrame, metric: str) -> pd.Series:
    """A metric with each signal's own level removed, keeping the model means.

    Signals differ far more in difficulty than models do, so a standard error
    over signals mostly measures the signals. Subtracting each signal's mean
    over models at every evaluation, and adding back the overall mean, leaves
    each model's mean curve unchanged while its spread shows only how
    consistently it scores relative to the others: the within-subject standard
    error of Cousineau (2005), with the correction of Morey (2008).
    """
    step = ["signal", "iteration"]
    cell = [data["model"], data["iteration"]]
    centered = (
        data[metric]
        - data.groupby(step)[metric].transform("mean")
        + data.groupby("iteration")[metric].transform("mean")
    )
    mean = centered.groupby(cell).transform("mean")
    models = data["model"].nunique()
    correction = (models / (models - 1)) ** 0.5 if models > 1 else 1.0
    return mean + (centered - mean) * correction


def convergence(
    data: pd.DataFrame,
    metrics: Sequence[str],
    out_dir: Path,
    models: Sequence[str] | None = None,
    time_limit: float | None = None,
) -> None:
    """Plot each metric against iteration and against time, averaged over signals.

    The band is one within-signal standard error (see :func:`within_signal`).
    Each evaluation is placed at the elapsed time averaged over signals, so
    that seaborn groups the signals of an evaluation together. ``time_limit``
    crops the time axis, where the slowest models would squeeze the rest.
    """
    models = list(models or order(data, metrics[0]))
    palette = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}
    dashes = {m: DASHES[i // len(PALETTE)] for i, m in enumerate(models)}
    data = data.assign(
        time=data.groupby(["model", "iteration"])["time"].transform("mean")
    )

    for metric in metrics:
        banded = data.assign(**{metric: within_signal(data, metric)})
        for axis, label in (("iteration", "Iteration"), ("time", "Time (s)")):
            figure, plot = plt.subplots()
            sns.lineplot(
                data=banded,
                x=axis,
                y=metric,
                hue="model",
                hue_order=models,
                palette=palette,
                style="model",
                dashes=dashes,
                errorbar=("se", 1),
                err_kws={"alpha": 0.1, "linewidth": 0},
                ax=plot,
            )
            plot.set(xlabel=label, ylabel=METRICS[metric][0])
            means = data.groupby(["model", "iteration"], as_index=False)[
                [axis, metric]
            ].mean()
            means = means.rename(columns={metric: "mean"})
            if axis == "time":
                plot.set_xlim(0, time_limit)
                if time_limit:
                    means = means[means["time"] <= time_limit]
            plot.set_ylim(*y_range(means, axis, METRICS[metric][1]))

            # Worst at the top, best at the bottom, where the curves leave room.
            handles, labels = plot.get_legend_handles_labels()
            plot.legend(
                handles[::-1],
                labels[::-1],
                loc="best",
                ncol=2 if len(models) > 8 else 1,
                framealpha=0.9,
                edgecolor="0.8",
            )
            save(figure, out_dir / f"{metric}_{axis}")


def model_config(run: dict[str, Any]) -> dict[str, Any]:
    """The config a run was trained with, for rebuilding its model."""
    recorded = run.get("config") or [f"configs/{run['task']}.yaml"]
    config: dict[str, Any] = {}
    for path in (Path(path) for path in recorded if "=" not in path):  # drop --set
        # Runs from another checkout recorded its absolute path, so fall back
        # to the same file here.
        local = ROOT / path
        if not local.exists():
            local = ROOT / "configs" / path.name
        merge(config, yaml.safe_load(local.read_text()))
    return config


def load_model(run: dict[str, Any], in_features: int, out_features: int, device):
    """Rebuild a run's model and load its checkpoint."""
    model_class, kwargs = build(run["setup"])
    model = model_class(in_features=in_features, out_features=out_features, **kwargs)
    state = torch.load(run["directory"] / "checkpoint.pt", map_location="cpu")
    model.load_state_dict(state)
    return model.to(device)


def figure_name(rank: int, model: str, metric: str, value: float) -> str:
    """``03_FUTON-sinc_36p85dB``: rank, model and score, for sorting by quality.

    The decimal point is written as ``p``: a dot would read as a file suffix,
    and the mesh and image writers replace one when they add their own.
    """
    unit = {"psnr": "dB", "iou": "iou"}.get(metric, metric)
    return f"{rank:02d}_{model}_{value:.2f}{unit}".replace(".", "p", 1)


def examples(
    runs: list[dict[str, Any]], signals: Sequence[str], metric: str
) -> dict[str, list[dict[str, Any]]]:
    """Group the runs of each requested signal, best first."""
    chosen = {}
    for signal in signals:
        matching = [run for run in runs if run["data"] == signal]
        if not matching:
            raise SystemExit(f"no runs for {signal!r}")
        chosen[signal] = sorted(
            matching,
            key=lambda run: scale(metric, run["metrics"][metric]),
            reverse=METRICS[metric][1],
        )
    return chosen


def montage(panels: Sequence[tuple[str, Path]], path: Path, columns: int = 5) -> None:
    """Lay saved images out as one figure, for a paper's qualitative row."""
    rows = -(-len(panels) // columns)
    width = plt.rcParams["figure.figsize"][0]
    figure, axes = plt.subplots(
        rows, columns, figsize=(width, width / columns * rows * 1.15), squeeze=False
    )
    for axis, (label, image) in zip(axes.ravel(), panels):
        axis.imshow(plt.imread(image))
        axis.set_title(label, fontsize=7, pad=2)
    for axis in axes.ravel():
        axis.set_axis_off()
    # PDF only: a PGF of raster panels writes each one out beside it.
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def style() -> None:
    """One look for every report: a figure that suits a paper column.

    Sizes are the final ones, since a PGF figure is included as it is. With
    ``pgf.rcfonts`` off the text is typeset in the document's own font, so the
    figures match the surrounding paper rather than carrying matplotlib's.
    """
    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update(
        {
            "figure.figsize": (5.5, 3.4),
            "figure.constrained_layout.use": True,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 7,
            "legend.borderpad": 0.4,
            "legend.columnspacing": 1.0,
            "legend.handlelength": 1.6,
            "lines.linewidth": 1.5,
            "pgf.texsystem": "pdflatex",
            "pgf.rcfonts": False,
            "savefig.pad_inches": 0.02,
        }
    )


def report(
    runs: list[dict[str, Any]],
    metrics: Sequence[str],
    out_dir: Path,
    task: str = "image",
    per_signal: bool = True,
    time_limit: float | None = None,
) -> pd.DataFrame:
    """Write the table and the convergence plots of a set of runs."""
    style()
    data = results(runs)
    summary = table(data, metrics, out_dir, SPEED[task], per_signal)
    convergence(curves(runs), metrics, out_dir, list(summary.index), time_limit)
    return summary
