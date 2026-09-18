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
STYLES = ("-", "--", ":")
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
    return f"{name} {'\u2191' if larger_is_better else '\u2193'}"


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
    except (RuntimeError, FileNotFoundError) as error:  # PGF needs LaTeX
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
    name: str = "summary",
) -> pd.DataFrame:
    """One table of every model: size, time, speed, and each metric.

    The unlabelled first columns hold the size, the times, and each metric
    averaged over the signals as ``mean+-std``; after them one super column
    per signal holds that signal's metrics. CSV and LaTeX keep those two
    header levels; Markdown, which has no multi-column header, joins them.
    """
    models = order(data, metrics[0])
    signals = sorted(data["signal"].unique())
    grouped = data.groupby("model")

    columns: list[tuple[str, str]] = []
    cells: dict[str, list[str]] = {model: [] for model in models}

    def add(column: tuple[str, str], values: dict[str, str]) -> None:
        columns.append(column)
        for model in models:
            cells[model].append(values.get(model, ""))

    add(
        ("", "# Params (k) \u2193"),
        {m: f"{grouped.get_group(m)['parameters'].iloc[0]:.1f}" for m in models},
    )
    add(
        ("", "Time (s) \u2193"),
        {m: f"{grouped.get_group(m)['time'].mean():.1f}" for m in models},
    )
    add(
        ("", f"Speed ({speed}) \u2191"),
        {m: f"{grouped.get_group(m)['speed'].mean():.1f}" for m in models},
    )
    for metric in metrics:  # averaged over the signals, so unlabelled
        add(
            ("", header(metric)),
            {
                m: entry(
                    metric,
                    grouped.get_group(m)[metric].mean(),
                    grouped.get_group(m)[metric].std(),
                )
                for m in models
            },
        )
    for signal in signals if per_signal else ():
        values = data[data["signal"] == signal].set_index("model")
        for metric in metrics:
            add(
                (signal, header(metric)),
                {
                    m: entry(metric, values[metric][m])
                    for m in models
                    if m in values.index
                },
            )

    summary = pd.DataFrame(
        [cells[model] for model in models],
        index=pd.Index(models),
        columns=pd.MultiIndex.from_tuples(columns),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / f"{name}.csv", index_label="Model")
    (out_dir / f"{name}.md").write_text(markdown(summary))
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
    (out_dir / f"{name}.tex").write_text(latex)
    return summary


def convergence(
    data: pd.DataFrame,
    metrics: Sequence[str],
    out_dir: Path,
    models: Sequence[str] | None = None,
    time_limit: float | None = None,
) -> None:
    """Plot each metric against iteration and against time, averaged over signals.

    The band is the standard error of the mean over the signals: a standard
    deviation would show the spread between signals, which is wide enough to
    bury the difference between models. The lines are drawn from the
    per-iteration mean rather than by seaborn's own aggregation, because the
    elapsed time differs between signals. ``time_limit``
    crops the time axis, where the slowest models would otherwise set a scale
    on which the rest finish immediately.
    """
    models = models or order(data, metrics[0])
    lines = {
        model: (PALETTE[index % len(PALETTE)], STYLES[index // len(PALETTE)])
        for index, model in enumerate(models)
    }
    for metric in metrics:
        grouped = data.groupby(["model", "iteration"]).agg(
            time=("time", "mean"),
            mean=(metric, "mean"),
            sd=(metric, "sem"),  # standard error: the deviation over sqrt(n)
        )
        for axis, label in (("iteration", "Iteration"), ("time", "Time (s)")):
            figure, plot = plt.subplots(figsize=(7, 4.5))
            for model in models:
                line = grouped.loc[model].reset_index()
                x = line["iteration"] if axis == "iteration" else line["time"]
                color, style = lines[model]
                plot.plot(
                    x,
                    line["mean"],
                    label=model,
                    color=color,
                    linestyle=style,
                    linewidth=2,
                )
                if line["sd"].notna().any():
                    plot.fill_between(
                        x,
                        line["mean"] - line["sd"],
                        line["mean"] + line["sd"],
                        color=color,
                        alpha=0.15,
                        linewidth=0,
                    )
            plot.set(xlabel=label, ylabel=METRICS[metric][0])
            if axis == "time":
                plot.set_xlim(0, time_limit)

            # Bands, early transients and a diverged model each set a range
            # on which the rest sit on top of each other. So keep the best end
            # and trim the weak one to the tenth percentile of the second half
            # of the run, where the models have converged and separate.
            shown = grouped.reset_index()
            if axis == "time" and time_limit:
                shown = shown[shown[axis] <= time_limit]
            converged = shown[shown[axis] >= shown[axis].median()]["mean"]
            if METRICS[metric][1]:
                low, high = converged.quantile(0.10), shown["mean"].max()
            else:
                low, high = shown["mean"].min(), converged.quantile(0.90)
            margin = 0.05 * (high - low)
            plot.set_ylim(low - margin, high + margin)

            # Worst at the top, best at the bottom, in the corner the rising
            # (or falling) curves leave free.
            handles, labels = plot.get_legend_handles_labels()
            plot.legend(
                handles[::-1],
                labels[::-1],
                loc="lower right" if METRICS[metric][1] else "upper right",
                ncol=2 if len(models) > 8 else 1,
                fontsize=8,
                framealpha=0.85,
            )
            save(figure, out_dir / f"{metric}_{axis}")


def model_config(run: dict[str, Any]) -> dict[str, Any]:
    """The config a run was trained with, for rebuilding its model."""
    recorded = run.get("config") or [f"configs/{run['task']}.yaml"]
    config: dict[str, Any] = {}
    for path in (path for path in recorded if "=" not in path):  # drop --set
        merge(config, yaml.safe_load((ROOT / path).read_text()))
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


def style() -> None:
    """One look for every report."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    plt.rcParams["figure.constrained_layout.use"] = True


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
