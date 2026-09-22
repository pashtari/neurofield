"""The paper's figures and tables, built from the runs under logs/.

Usage:
    python scripts/report_paper.py
    python scripts/report_paper.py --tasks image --overwrite
    python scripts/report_paper.py --no-panels

Writes results/, which holds the paper's figures and tables and nothing else:
    legend.{pdf,pgf}                 the models' legend, shared by every figure
    <task>/convergence.{pdf,pgf}     quality against training time, for FUTON and
                                     the strongest model of each other family
    <task>/tradeoff.{pdf,pgf}        every model's final quality against its
                                     training time
    <task>/throughput.{pdf,pgf}      the same against its inference rate
    <task>/table.{tex,md}            every model's size, training time and final
                                     metrics, the averages with a paired standard
                                     error: occupancy adds every shape's IoU, and
                                     NeRF every scene's metrics under a super column
    <task>/qualitative_<signal>.pdf  the signal with two regions boxed, and those
                                     regions magnified for the ground truth and
                                     every featured model
    <task>/panels/<signal>/          the renders those magnifications are cut from,
                                     rebuilt from the checkpoints and kept, so that
                                     recomposing a figure needs no GPU
    ablation/basis.{tex,md}          every basis at the benchmark's size
    ablation/components.{tex,md}     the combiner and the decoder, for both bases
    ablation/rank_<basis>.{pdf,pgf}  IoU against the CP rank at each K

Tables mark the best value in bold and the second underlined. Drawing the panels
needs the signals in data/ and a GPU; --no-panels leaves them out, and
--overwrite draws them again, which a changed NeRF view calls for.

The magnified regions are found, and a run prints the centre of every one it
drew, ready to paste into CENTERS. Setting a signal's centres there fixes them
by hand, in fractions of the trimmed reference, and SIDES sets how large they
are. Panels already drawn are reused, so moving a box and running again costs
seconds and no GPU.
"""

import argparse
import json
import re
import warnings
from collections.abc import Sequence
from itertools import groupby
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from matplotlib.axis import Axis
from matplotlib.backends.backend_pgf import LatexError
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from matplotlib.patches import Rectangle
from matplotlib.ticker import (
    FixedLocator,
    MaxNLocator,
    NullFormatter,
    NullLocator,
    StrMethodFormatter,
)
from PIL import Image
from train_common import ROOT, build, merge

import neurofield as nf

# Header, whether larger is better, the factor the stored value is shown in,
# and the decimals of a table entry.
METRICS = {
    "psnr": ("PSNR (dB)", True, 1, 2),
    "ssim": ("SSIM (%)", True, 100, 2),
    "lpips": ("LPIPS", False, 1, 4),
    "iou": ("IoU (%)", True, 100, 2),
}
SPEED = {"image": "img/s", "occupancy": "vol/s", "nerf": "views/s"}
SIGNALS = {  # where a task's signal lives
    "image": "data/Kodak/{}.png",
    "occupancy": "data/occupancy/{}.ply",
    "nerf": "data/nerf/blender/{}",
}
# Per task, the metric plotted, its axis label, and the metrics tabulated.
TASKS = {
    "image": ("psnr", "PSNR (dB)", ("psnr", "ssim", "lpips")),
    "occupancy": ("iou", "IoU (%)", ("iou",)),
    "nerf": ("psnr", "PSNR (dB)", ("psnr", "ssim", "lpips")),
}
EXAMPLES = {  # the two signals each task shows, the ones FUTON gains most on
    "image": ("kodim01", "kodim17", "kodim19", "kodim21"),
    "occupancy": ("thai_statue", "armadillo"),
    "nerf": ("lego", "hotdog", "materials"),
}
# The test view a scene is rendered from, the one where FUTON gains most of
# those that show the scene whole: every run records all 200, so the choice is
# made from the logs rather than by eye.
VIEWS = {"lego": 184, "hotdog": 136, "materials": 86}
# Where a magnification may be taken, for the signals whose telling regions the
# search would pass over: one rectangle per box, as fractions of the trimmed
# reference (left, top, right, bottom). The best square inside each is still
# the one found.
ZONES = {}
# Centres set by hand, one per magnified region, as fractions of the trimmed
# reference. A signal named here takes these instead of searching, so this is
# where to put a region the search will not find. Running the report prints the
# centres it used for every signal, in this format, to start from.
CENTERS: dict[str, tuple[tuple[float, float], ...]] = {
    "kodim01": ((0.904, 0.106), (0.592, 0.323)),
    "kodim17": ((0.67, 0.22), (0.16, 0.63)),
    "kodim19": ((0.67, 0.22), (0.25, 0.67)),
    "kodim21": ((0.15, 0.54), (0.5, 0.6)),
    "thai_statue": ((0.35, 0.8), (0.77, 0.92)),
    "armadillo": ((0.5, 0.5), (0.3, 0.8)),
    "lego": ((0.356, 0.260), (0.383, 0.678)),
    "hotdog": ((0.475, 0.210), (0.796, 0.333)),
    "materials": ((0.407, 0.639), (0.667, 0.741)),
}
UP = {"lucy": 2}  # the Stanford Lucy is z up; the other four shapes are y up

# Categorical hues in a fixed order, and a light-to-dark blue ramp for ordered
# levels. Four hues are as many as stay apart under every kind of colour
# blindness, so a hue names a family and the line style names the member.
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7")
RAMP = ("#9ec5f4", "#5598e7", "#2a78d6", "#184f95", "#0d366b")
# FUTON and the baselines, by family: periodic activations (SIREN and FINER,
# which is SIREN with a variable-periodic one), hash grids (Instant-NGP) and
# grid factorizations (TensoRF). A hue names the family and the line style the
# member, the stronger of a pair solid. Colour, dashes, and a filled marker.
FEATURED = {
    "SIREN": (PALETTE[1], (4, 2), False),
    "FINER": (PALETTE[1], "", True),
    "Instant-NGP": (PALETTE[2], "", True),
    "TensoRF": (PALETTE[3], "", True),
    "FUTON-sinc": (PALETTE[0], "", True),
    "FUTON-lanczos": (PALETTE[0], (4, 2), False),
}
OTHERS = "0.7"  # gray of every other model
# TensoRF is its default variant for the dimension: CP in 2D, where it is the
# only one, and VM in 3D, the one its authors recommend.
TENSORF = {"image": "TensoRF", "occupancy": "TensoRF-VM", "nerf": "TensoRF-VM"}
ROWS = (
    "RFF", "PE-MLP", "MFN", "SIREN", "Gauss", "WIRE", "FINER", "Instant-NGP",
    "TensoRF", "GA-Planes", "FUTON-sinc", "FUTON-lanczos",
)  # fmt: skip
# Metric axes are linear, which needs no explaining; IoU saturates, so its
# curves start here instead, which the ticks show plainly.
FLOOR = {"occupancy": 99.0}
TARGET = 99.8  # IoU (%) whose first time the text quotes
SIZE = (2.3, 1.9)  # inches, a third of a two-column page with gutters
WIDTH = 7.0  # inches, a two-column figure

# One colour per magnified region, so that a box and the row it magnifies are
# read together: the red of the super-resolution literature, and a cyan that
# keeps its distance from it under every kind of colour blindness.
COLORS = ("#e02020", "#00a2c7")
# The side of a magnified region, in fractions of the reference's geometric
# mean, which sizes it alike whatever the aspect ratio. The regions themselves
# are found, not set: see :func:`regions`. A NeRF view is 200 pixels across, so
# its regions hold fewer of them and stay wider.
SIDES = {"image": 0.075, "occupancy": 0.14, "nerf": 0.20}
# The magnifications' titles and scores: the largest size they are set at, and
# the inches kept above the panels for the titles and below for the scores. A
# talk raises both before drawing; the paper keeps them.
LABEL_SIZE = 6.5
LABEL_ROOM = (0.17, 0.19)
# The widest a reference is drawn, as an aspect ratio: a wider one is scaled
# down and centred on the rows, so that a panorama does not squeeze the
# magnifications. None draws every reference at the full height of the rows,
# as the paper does; the deck caps it.
REFERENCE_ASPECT: float | None = None


def text_width(text: str, weight: str = "normal") -> float:
    """The width of ``text`` at a font size of 1pt, in points, in the style's font."""
    path = TextPath((0, 0), text, size=1, prop=FontProperties(weight=weight))
    return path.get_extents().width


UNITS = {"psnr": "{:.2f} dB", "iou": "{:.2f}%"}

# The FUTON ablations, all on the occupancy task: the bases from the smoothest
# to the most local, and the variants of the two stages after the basis, both
# for the bases the benchmark uses.
STUDIES = ("basis", "components_rank", "tensor_net", "decoder")
GRID = re.compile(r"FUTON-(?P<basis>\w+)-K(?P<components>\d+)-R(?P<rank>\d+)$")
BASES = ("Cosine", "Chebyshev", "Legendre", "Triangle", "Lanczos", "Sinc")
PAIR = ("sinc", "lanczos")
VARIANTS = {  # row: the model it stands for, in whichever study ran it
    "CP, MLP": "FUTON-{basis}",
    "TR, MLP": "FUTON-{basis}-TR",
    "CP, linear (R = 342)": "FUTON-{basis}-linear",
    "CP, linear (R = 218)": "FUTON-{basis}-linear-R218",
}
PARAMS = ("", "# Params (k)", None, 1)  # the column every table opens with
DASHES = ("", (4, 2), (1, 1.5))  # solid, dashed, dotted
# The convergence of the two studies that get a curve, as label, colour and
# dashes per model: a hue names the choice under study and the dashes the basis
# it is made with, but for the bases themselves, where the hue separates the
# compactly supported ones from those that span the axis. The decoder study
# keeps to its table, and the rank study to its figures.
CURVES = {
    "basis": {
        "FUTON-cosine": ("Cosine", PALETTE[1], DASHES[0]),
        "FUTON-chebyshev": ("Chebyshev", PALETTE[1], DASHES[1]),
        "FUTON-legendre": ("Legendre", PALETTE[1], DASHES[2]),
        "FUTON-triangle": ("Triangle", PALETTE[0], DASHES[2]),
        "FUTON-lanczos": ("Lanczos", PALETTE[0], DASHES[1]),
        "FUTON-sinc": ("Sinc", PALETTE[0], DASHES[0]),
    },
    "tensor_net": {
        "FUTON-sinc": ("CP, sinc", PALETTE[0], DASHES[0]),
        "FUTON-lanczos": ("CP, lanczos", PALETTE[0], DASHES[1]),
        "FUTON-sinc-TR": ("TR, sinc", PALETTE[2], DASHES[0]),
        "FUTON-lanczos-TR": ("TR, lanczos", PALETTE[2], DASHES[1]),
    },
}


def style() -> None:
    """One look for every figure.

    Sizes are final, since a PGF figure is included as it is, and with
    ``pgf.rcfonts`` off the text takes the document's own font.
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
            "legend.handlelength": 1.6,
            "lines.linewidth": 1.5,
            "pgf.texsystem": "pdflatex",
            "pgf.rcfonts": False,
            "savefig.pad_inches": 0.02,
        }
    )


def save(figure: plt.Figure, path: Path) -> None:
    """Write a figure as PDF, and as PGF to include in a LaTeX document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    try:
        figure.savefig(path.with_suffix(".pgf"), bbox_inches="tight")
    except (RuntimeError, FileNotFoundError, LatexError) as error:  # PGF runs LaTeX
        warnings.warn(f"no {path.name}.pgf: {error}")
    plt.close(figure)


def read(log_dir: Path, task: str) -> list[dict[str, Any]]:
    """Load the ``task`` runs under ``log_dir``, each tagged with its directory."""
    runs = []
    for path in sorted(log_dir.rglob("results.json")):
        run = json.loads(path.read_text())
        if run["task"] == task:
            runs.append(run | {"directory": path.parent})
    if not runs:
        raise SystemExit(f"no {task} runs under {log_dir}")
    return runs


def scale(metric: str, value: float) -> float:
    """A stored value as it is reported, e.g. IoU and SSIM as percentages."""
    return value * METRICS[metric][2]


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


def load(task: str) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    """A task's runs, its final metrics and its curves, with one TensoRF."""
    others = {"TensoRF", "TensoRF-CP", "TensoRF-VM"} - {TENSORF[task]}
    runs = [
        run | {"model": "TensoRF" if run["model"] == TENSORF[task] else run["model"]}
        for run in read(ROOT / "logs" / task, task)
        if run["model"] not in others
    ]
    return runs, results(runs), curves(runs)


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


def y_range(means: pd.DataFrame, metric: str) -> tuple[float, float]:
    """A range on which the models separate, from their mean curves.

    Early transients would squeeze the rest together, so the weak end is
    trimmed to the tenth percentile of the second half of the run, then
    widened to every final value but a collapsed one.
    """
    converged = means[means["iteration"] >= means["iteration"].median()][metric]
    finals = means.groupby("model")[metric].last()
    finals = finals[finals >= 0.5 * finals.max()]
    low, high = min(converged.quantile(0.10), finals.min()), means[metric].max()
    margin = 0.05 * (high - low)
    return low - margin, high + margin


def paired_error(final: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    """Each model's standard error over signals, once their levels are removed.

    Signals differ far more than models do, so the plain deviation over them
    would say how hard the signals are; see :func:`within_signal`.
    """
    final = final.assign(iteration=0)  # one evaluation, the last
    return pd.DataFrame(
        {
            ("mean", metric): final.assign(**{metric: within_signal(final, metric)})
            .groupby("model")[metric]
            .sem()
            for metric in metrics
        }
    )


def log_ticks(axis: Axis) -> None:
    """Label a log axis 1, 2, 5, 10, ... rather than 10^0, at most five ticks.

    Of the 1-2-3-5, 1-2-5 and 1 ticks per decade, it takes the densest that
    keeps to five within the axis' limits.
    """
    low, high = axis.get_view_interval()
    decades = range(int(np.log10(low)) - 1, int(np.log10(high)) + 2)
    for subs in ((1, 2, 3, 5), (1, 2, 5), (1,)):
        ticks = [
            m * 10.0**k for k in decades for m in subs if low <= m * 10.0**k <= high
        ]
        if len(ticks) <= 5:
            break
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(StrMethodFormatter("{x:g}"))
    axis.set_minor_formatter(NullFormatter())


def padded(values: pd.Series, fraction: float = 0.08) -> tuple[float, float]:
    """Limits that leave a margin, so that no marker is clipped by a spine."""
    low, high = values.min(), values.max()
    margin = fraction * (high - low) or abs(low) * fraction or 1.0
    return low - margin, high + margin


def legend() -> plt.Figure:
    """The featured models' lines and markers, and the other models' gray dot."""
    handles = [
        Line2D([], [], color=color, dashes=dashes or (None, None), marker="o",
               markerfacecolor=color if filled else "white", label=model)
        for model, (color, dashes, filled) in FEATURED.items()
    ] + [Line2D([], [], color=OTHERS, marker="o", ls="", label="Other models")]  # fmt: skip
    figure = plt.figure(figsize=(3 * SIZE[0], 0.3))
    figure.legend(handles=handles, loc="center", ncol=len(handles), frameon=False)
    return figure


def convergence(curve: pd.DataFrame, task: str) -> plt.Figure:
    """The task's metric against training time, for the featured models.

    Time is logarithmic, since the curves span more than a decade of it.
    """
    metric, label, _ = TASKS[task]
    curve = curve[curve["model"].isin(FEATURED)]
    curve = curve.assign(
        time=curve.groupby(["model", "iteration"])["time"].transform("mean"),
        **{metric: within_signal(curve, metric)},
    )
    figure, plot = plt.subplots(figsize=SIZE)
    sns.lineplot(
        data=curve,
        x="time",
        y=metric,
        hue="model",
        palette={model: color for model, (color, _, _) in FEATURED.items()},
        style="model",
        dashes={model: dashes for model, (_, dashes, _) in FEATURED.items()},
        errorbar=("se", 1),
        err_kws={"alpha": 0.15, "linewidth": 0},
        legend=False,
        ax=plot,
    )
    # A range on which the models separate, from the first evaluation inside
    # it to the last.
    means = curve.groupby(["model", "iteration"], as_index=False)[
        [metric, "time"]
    ].mean()
    low, high = y_range(means, metric)
    low = max(low, FLOOR.get(task, low))
    shown = means[means[metric].between(low, high)]["time"]
    # The marker the legend carries, at the end of each curve.
    for model, (color, _, filled) in FEATURED.items():
        last = means[means["model"] == model].iloc[-1]
        plot.plot(last["time"], last[metric], marker="o", markersize=4,
                  color=color, markerfacecolor=color if filled else "white",
                  clip_on=False, zorder=4)  # fmt: skip
    plot.set(
        xscale="log",
        xlim=(shown.min() / 1.1, means["time"].max() * 1.1),
        ylim=(low, high),
        xlabel="Training time (s)",
        ylabel=label,
    )
    log_ticks(plot.xaxis)
    return figure


def tradeoff(final: pd.DataFrame, task: str, cost: str = "time") -> plt.Figure:
    """Each model's final quality against its training time or inference rate.

    Both axes are linear: the models lie within a decade of each other, where
    a log axis would only cost the reader the plain reading of the distances.
    """
    metric, label, _ = TASKS[task]
    # The cost axis carries its direction, as the table headers do: a reader
    # pauses over whether more inference rate is better, never over PSNR.
    axis = {
        "time": r"Training time (s)$\,\downarrow$",
        "speed": rf"Inference ({SPEED[task]})$\,\uparrow$",
    }[cost]
    means = final.groupby("model")[[cost, metric]].mean()
    others = means[~means.index.isin(FEATURED)]
    figure, plot = plt.subplots(figsize=SIZE)
    plot.scatter(others[cost], others[metric], s=14, color=OTHERS, zorder=2)
    for model, (color, _, filled) in FEATURED.items():
        x, y = means.loc[model, [cost, metric]]
        plot.scatter(
            x,
            y,
            s=30,
            facecolor=color if filled else "white",
            edgecolor=color,
            zorder=3,
        )
    # Explicit limits, so that the pair of figures shares a quality axis and
    # no marker touches a spine.
    plot.set(
        xlim=padded(means[cost]),
        ylim=padded(means[metric]),
        xlabel=axis,
        ylabel=label,
    )
    plot.xaxis.set_major_locator(MaxNLocator(5))
    plot.yaxis.set_major_locator(MaxNLocator(5))
    return figure


def table(
    values: pd.DataFrame,
    spec: dict,
    errors: pd.DataFrame | None = None,
    order: Sequence[str] = ROWS,
    index: str = "Model",
    highlight: bool = False,
) -> dict[str, str]:
    """LaTeX and Markdown of models by columns, best bold and second underlined.

    ``spec`` maps each column of ``values`` to its group (empty for none), its
    header, whether larger is better (``None`` not to rank it) and its
    decimals; values tied at those decimals share a rank. A column of
    ``errors`` follows its value as ``mean+-error``. Consecutive columns of one
    group share a spanning header. Rows follow ``order``, which the ablations
    give their own, and ``index`` heads them. ``highlight`` tints the two best
    cells of every ranked column, which reads from the back of a room but would
    be noise in a printed table.
    """
    models = [model for model in order if model in values.index]
    cells = {}
    for name, (_, _, larger_is_better, decimals) in spec.items():
        shown = values[name].reindex(models).round(decimals)
        error = None if errors is None or name not in errors else errors[name]
        ranks = shown.rank(ascending=not larger_is_better, method="min")
        for model in models:
            text = "–" if pd.isna(shown[model]) else f"{shown[model]:.{decimals}f}"
            spread = (
                f"{error[model]:.{decimals}f}"
                if error is not None and not pd.isna(shown[model])
                else ""
            )
            rank = ranks[model] if larger_is_better is not None else np.nan
            cells[model, name] = text, spread, rank

    def line(model: str, mark: dict[int, str], spread: str) -> list[str]:
        # The mean carries the emphasis; the error follows it, set smaller, so
        # that a column of values reads down the page before it reads across.
        return [model] + [
            mark.get(rank, "{}").format(text) + (spread.format(error) if error else "")
            for text, error, rank in (cells[model, name] for name in spec)
        ]

    titles = [index] + [
        title if larger_is_better is None
        else f"{title} {'↑' if larger_is_better else '↓'}"
        for _, title, larger_is_better, _ in spec.values()
    ]  # fmt: skip
    groups = [""] + [group for group, *_ in spec.values()]
    # Numbers right against a fixed number of decimals, which lines their
    # points up; the headers stay centred over them.
    latex = [rf"\begin{{tabular}}{{l{'r' * (len(titles) - 1)}}}", r"\toprule"]
    flat = [f"{group} {title}".strip() for group, title in zip(groups, titles)]
    if any(groups):  # a spanning header over each group's columns
        spans, rules, first = [], [], 1
        for group, columns in groupby(groups):
            width = len(list(columns))
            spans.append(rf"\multicolumn{{{width}}}{{c}}{{{group}}}")
            if group:
                rules.append(rf"\cmidrule(lr){{{first}-{first + width - 1}}}")
            first += width
        latex += [" & ".join(spans) + r" \\", "".join(rules)]
    centered = [titles[0]] + [rf"\multicolumn{{1}}{{c}}{{{t}}}" for t in titles[1:]]
    latex += [" & ".join(centered) + r" \\", r"\midrule"]
    markdown = ["| " + " | ".join(flat) + " |", "| --- " * len(flat) + "|"]
    tex_mark = (
        {1: r"\cellcolor{{best}}\textbf{{{}}}", 2: r"\cellcolor{{second}}{}"}
        if highlight
        else {1: r"\textbf{{{}}}", 2: r"\underline{{{}}}"}
    )
    for model in models:
        if model == "FUTON-sinc":
            latex.append(r"\midrule")
        latex.append(
            " & ".join(line(model, tex_mark, r"{{\scriptsize$\pm${}}}")) + r" \\"
        )
        markdown.append(
            "| " + " | ".join(line(model, {1: "**{}**", 2: "_{}_"}, "±{}")) + " |"
        )
    latex += [r"\bottomrule", r"\end{tabular}"]
    tex = "\n".join(latex) + "\n"
    # None of the commands above holds one of these, so escaping is safe.
    for symbol, command in (
        ("↑", r"$\uparrow$"),
        ("↓", r"$\downarrow$"),
        ("%", r"\%"),
        ("#", r"\#"),
        ("_", r"\_"),
    ):
        tex = tex.replace(symbol, command)
    return {"tex": tex, "md": "\n".join(markdown) + "\n"}


def per_signal(
    final: pd.DataFrame, metrics: Sequence[str]
) -> tuple[pd.DataFrame, dict]:
    """Each signal's metrics and their mean, as values and columns.

    Several metrics group the columns by signal and name the metrics inside
    them; one metric is itself the group, over a column per signal.
    """
    signals = sorted(final["signal"].unique())
    per = {
        metric: final.pivot_table(index="model", columns="signal", values=metric)
        for metric in metrics
    }
    values, spec = {}, {}
    for signal in [*signals, "mean"]:  # a signal's metrics stay side by side
        title = "Mean" if signal == "mean" else signal.replace("_", " ").title()
        for metric in metrics:
            name, larger_is_better, _, decimals = METRICS[metric]
            column = (
                per[metric][signals].mean(axis=1)
                if signal == "mean"
                else per[metric][signal]
            )
            group, header = (
                (title, name.split(" (")[0]) if len(metrics) > 1 else (name, title)
            )
            values[signal, metric] = column
            spec[signal, metric] = (group, header, larger_is_better, decimals)
    frame = pd.DataFrame(values)
    # Plain tuples, not a column MultiIndex, so that they join the flat
    # columns beside them and key the spec.
    frame.columns = pd.Index(list(values), tupleize_cols=False)
    return frame, spec


def tables(final: pd.DataFrame, task: str) -> dict[str, str]:
    """The task's table, with its metrics per signal where the signals are few.

    The averaged columns carry one paired standard error over the signals.
    NeRF's table has a super column per scene over its three metrics, so it
    needs the width of a page turned sideways.
    """
    metrics = TASKS[task][2]
    # The size and training time every table opens with, as TensoRF, K-Planes
    # and NeuRBF report them; the inference rate stays in the throughput plot.
    spec = {"parameters": PARAMS, "time": ("", "Train time (s)", False, 1)}
    values = final.groupby("model")[list(spec)].mean()
    errors = paired_error(final, metrics)
    if task == "image":  # 24 images, so their mean alone
        for metric in metrics:
            name, larger_is_better, _, decimals = METRICS[metric]
            spec[metric] = ("", name, larger_is_better, decimals)
        values = values.join(final.groupby("model")[list(metrics)].mean())
        errors.columns = list(metrics)
        return table(values, spec, errors)
    per, per_spec = per_signal(final, metrics)
    return table(values.join(per), spec | per_spec, errors)


def time_to_target(curve: pd.DataFrame) -> pd.Series:
    """Mean time each model first reaches the target IoU, if on every shape."""
    reached = curve[curve["iou"] >= TARGET].groupby(["model", "signal"])["time"]
    first = reached.min().groupby("model")
    return first.mean().where(first.size() == curve["signal"].nunique())


def render_image(signal: str, runs: list[dict], directory: Path, device) -> None:
    """The image itself, and each model's reconstruction of it."""
    dataset = nf.ImageCoordinateDataset(ROOT / SIGNALS["image"].format(signal))
    dataset.save(dataset.original, directory / "truth")
    for run in runs:
        model = load_model(run, 2, 3, device)
        output = nf.chunked_inference(model, dataset.input, device=device)
        dataset.save(dataset.postprocess(output.cpu()), directory / run["model"])


def render_occupancy(signal: str, runs: list[dict], directory: Path, device) -> None:
    """The shape's mesh, and each model's, drawn from one camera.

    Each run is checked against the shape it sees here, since a mesh file that
    differs from the one it was trained on would rebuild silently wrong.
    """
    resolution = model_config(runs[0])["data"]["resolution"]
    dataset = nf.OccupancyCoordinateDataset(
        ROOT / SIGNALS["occupancy"].format(signal),
        resolution=resolution,
        item_id=signal,
    )
    up = UP.get(signal, 1)
    dataset.save(dataset.original, directory / "truth", up=up, mesh=False)
    for run in runs:
        model = load_model(run, 3, 1, device)
        output = nf.chunked_inference(
            model, dataset.input, chunk_size=512**2, device=device
        )
        volume = dataset.postprocess(output.cpu())
        rebuilt = scale("iou", nf.iou(volume, dataset.original))
        recorded = scale("iou", run["metrics"]["iou"])
        if abs(rebuilt - recorded) > 0.5:
            raise SystemExit(
                f"{run['model']} rebuilds to {rebuilt:.2f}% IoU, not the "
                f"{recorded:.2f}% of its run: {signal} is not the shape it saw."
            )
        dataset.save(volume, directory / run["model"], up=up, mesh=False)


def field_and_renderer(run: dict, config: dict, dataset, device):
    """Rebuild a run's radiance field and its renderer, occupancy grid included.

    Only the field is checkpointed, and a fresh renderer starts fully
    occupied, so the grid is rebuilt from the field as training's warmup did.
    Without it, the faint density a model leaves in empty space, which the
    grid pruned when the run was scored, is sampled everywhere and rendered
    as fog: a sine network then looks far worse than its PSNR says.
    """
    field = nf.nerf.RadianceField(
        density_net=build(run["setup"]), aabb=dataset.aabb, **config["field"]
    )
    state = torch.load(run["directory"] / "checkpoint.pt", map_location="cpu")
    field.load_state_dict(state)
    renderer = nf.nerf.create_renderer(
        backend="nerfacc" if nf.nerf.is_nerfacc_available() else "pytorch",
        aabb=dataset.aabb,
        near=dataset.near,
        far=dataset.far,
    )
    field, renderer = field.to(device), renderer.to(device)
    with torch.no_grad():
        for step in range(renderer.warmup_steps + renderer.update_interval):
            renderer.update_occupancy(field, step)
    return field, renderer


def render_nerf(signal: str, runs: list[dict], directory: Path, device) -> None:
    """One test view of the scene, and each model's render of it."""
    config = model_config(runs[0])
    dataset = nf.nerf.BlenderDataset(
        ROOT / SIGNALS["nerf"].format(signal),
        "test",
        downsample=config["data"]["downsample"],
    )
    batch = dataset[VIEWS[signal]]
    target = (dataset.composite(batch["rgba"]) * 255).round().byte().numpy()
    Image.fromarray(target).save(directory / "truth.png")
    for run in runs:
        field, renderer = field_and_renderer(run, config, dataset, device)
        rendered = nf.nerf.render_image(
            field,
            renderer,
            batch["rays_o"].to(device),
            batch["rays_d"].to(device),
            background=dataset.background.to(device),
        )
        image = (rendered["rgb"].clamp(0, 1) * 255).round().byte().cpu().numpy()
        Image.fromarray(image).save(directory / f"{run['model']}.png")


def render_orbit(
    signal: str,
    runs: list[dict],
    directory: Path,
    device,
    frames: int,
    overwrite: bool = False,
) -> None:
    """An orbit of each model as a GIF, at the elevation the dataset uses.

    Nothing in the paper takes these; they are for a talk, and cost a minute
    a scene, so ``--orbit`` asks for them.
    """
    config = model_config(runs[0])
    dataset = nf.nerf.BlenderDataset(
        ROOT / SIGNALS["nerf"].format(signal),
        "test",
        downsample=config["data"]["downsample"],
    )
    poses = [
        nf.nerf.pose_spherical(theta, -30.0, dataset.radius)
        for theta in np.linspace(-180, 180, frames, endpoint=False)
    ]
    for run in runs:
        path = directory / f"{run['model']}.gif"
        if path.exists() and not overwrite:
            continue
        print(f"  orbiting {run['model']} on {signal}", flush=True)
        field, renderer = field_and_renderer(run, config, dataset, device)
        orbit = [
            Image.fromarray(frame)
            for frame in nf.nerf.render_views(field, renderer, dataset, poses)
        ]
        orbit[0].save(
            path,
            save_all=True,
            append_images=orbit[1:],
            duration=1000 // 15,
            loop=0,
        )


RENDER = {"image": render_image, "occupancy": render_occupancy, "nerf": render_nerf}


def panels(
    task: str, signal: str, runs: list[dict], device: str, overwrite: bool
) -> Path:
    """The signal and every featured model's reconstruction of it, as PNGs.

    Rendering rebuilds each model from its checkpoint, which needs the signal
    in data/ and a GPU, so the panels are kept and reused.
    """
    directory = ROOT / "results" / task / "panels" / signal
    missing = {
        model
        for model in FEATURED
        if overwrite or not (directory / f"{model}.png").exists()
    }
    if not missing and (directory / "truth.png").exists():
        return directory
    directory.mkdir(parents=True, exist_ok=True)
    chosen = [run for run in runs if run["data"] == signal and run["model"] in missing]
    if len(chosen) != len(missing):
        raise SystemExit(f"{len(chosen)} of {len(missing)} missing runs for {signal}")
    print(f"  rendering {len(chosen)} panels of {signal}", flush=True)
    RENDER[task](signal, chosen, directory, device)
    return directory


def trim(image: np.ndarray, margin: float = 0.02) -> tuple[slice, slice]:
    """The part of a render the signal occupies, plus a margin.

    Mesh and NeRF renders sit in a wide blank field that would leave the
    reference panel tiny. The box comes from the ground truth alone, so that
    every panel of a signal is cropped alike and the magnifications align.
    """
    ink = np.abs(image[..., :3] - image[0, 0, :3]).sum(-1) > 0.05
    rows, columns = (np.flatnonzero(ink.any(axis=axis)) for axis in (1, 0))
    if not rows.size or not columns.size:  # a photograph fills its frame
        return slice(None), slice(None)
    pad = int(margin * max(image.shape[:2]))
    return (
        slice(max(rows[0] - pad, 0), rows[-1] + pad),
        slice(max(columns[0] - pad, 0), columns[-1] + pad),
    )


def box_sums(field: np.ndarray, side: int) -> np.ndarray:
    """The sum over every ``side`` by ``side`` square, from an integral image."""
    total = np.pad(field, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return (
        total[side:, side:]
        - total[:-side, side:]
        - total[side:, :-side]
        + total[:-side, :-side]
    )


def detail(gray: np.ndarray, window: int = 5) -> np.ndarray:
    """How much fine detail the reference carries around each pixel.

    The energy left when a local mean is taken away, which is high on a fence,
    a lettering or a row of studs, and low on the long smooth edges whose blur
    costs the most error but tells a reader least.
    """
    pad = ((window // 2, window - 1 - window // 2),) * 2
    mean = np.pad(box_sums(gray, window) / window**2, pad, mode="edge")
    return (gray - mean) ** 2


def regions(
    panel: dict[str, np.ndarray],
    ours: str,
    side: int,
    zones: tuple | None = None,
) -> list[tuple[int, int]]:
    """The squares where the models differ most and ``ours`` is closest.

    A magnification earns its space only where the models disagree, so the
    regions are found rather than set: the spread of the panels about their
    mean, weighted by the detail the reference carries there and summed over
    every square of the given side. Spread favours no model, which asking
    where FUTON beats one chosen baseline would; the squares are then kept to
    those where FUTON is in fact the closest of the panels shown, so that the
    figure magnifies what it claims.

    A render sits on a blank field, where a silhouette a voxel out of place
    costs more error than any surface detail, so there the squares are kept
    inside the signal; a photograph fills its frame and keeps every square. The
    second square stands twice its side clear of the first where the signal
    leaves room for it, and merely clear of it where it does not. Given
    ``zones``, each square is instead the best one inside its own rectangle,
    which says where to look without saying what to take.
    """
    truth = panel["truth"]
    shown = [name for name in panel if name != "truth"]
    spread = np.var([panel[name].mean(-1) for name in shown], axis=0)
    scores = box_sums(spread * detail(truth.mean(-1)), side)

    error = {
        name: box_sums(((panel[name] - truth) ** 2).sum(-1), side) for name in shown
    }
    leads = error[ours] <= np.min([error[n] for n in shown if n != ours], axis=0)
    if leads.any():
        scores = np.where(leads, scores, -np.inf)
    blank = np.abs(truth - truth[0, 0]).sum(-1) <= 0.05
    if blank.mean() > 0.15:  # a render on a blank field, not a photograph
        inside = box_sums((~blank).astype(float), side) >= 0.75 * side**2
        scores = np.where(inside, scores, -np.inf)

    if zones:
        height, width = truth.shape[:2]
        found = []
        for left, top, right, bottom in zones:
            inside = np.full_like(scores, -np.inf)
            x, y = round(left * width), round(top * height)
            rows = slice(y, max(round(bottom * height) - side, y + 1))
            columns = slice(x, max(round(right * width) - side, x + 1))
            inside[rows, columns] = scores[rows, columns]
            row, column = np.unravel_index(np.argmax(inside), inside.shape)
            found.append((int(column), int(row)))
        return found

    top, left = np.unravel_index(np.argmax(scores), scores.shape)
    found = [(int(left), int(top))]
    for apart in (2.0, 1.0):  # room to breathe, or merely no overlap
        clear = scores.copy()
        keep = round(apart * side)
        clear[max(top - keep, 0) : top + keep, max(left - keep, 0) : left + keep] = (
            -np.inf
        )
        if np.isfinite(clear).any():
            row, column = np.unravel_index(np.argmax(clear), clear.shape)
            found.append((int(column), int(row)))
            break
    return found


def qualitative(task: str, signal: str, final: pd.DataFrame, directory: Path):
    """The reference with its regions boxed, and every panel's magnification.

    One row of magnifications per region, framed in the colour of its box, and
    the reference beside them spanning the rows, as wide as its aspect ratio
    makes it so that no panel is padded with white.
    """
    metric, _, _ = TASKS[task]
    scores = final[final["signal"] == signal].set_index("model")[metric]
    panel = {
        name: plt.imread(directory / f"{name}.png")[..., :3]
        for name in ["truth", *FEATURED]
    }
    box = trim(panel["truth"])
    panel = {name: image[box] for name, image in panel.items()}
    height, width = panel["truth"].shape[:2]

    ours = scores[["FUTON-sinc", "FUTON-lanczos"]].idxmax()
    side = round(SIDES[task] * np.sqrt(width * height))
    if signal in CENTERS:  # set by hand, in fractions of the trimmed reference
        found = [
            (
                min(max(round(x * width - side / 2), 0), width - side),
                min(max(round(y * height - side / 2), 0), height - side),
            )
            for x, y in CENTERS[signal]
        ]
    else:
        found = regions(panel, ours, side, ZONES.get(signal))
    centres = ", ".join(
        f"({(left + side / 2) / width:.3f}, {(top + side / 2) / height:.3f})"
        for left, top in found
    )
    print(f'    "{signal}": ({centres}),  # centres, for CENTERS', flush=True)

    rows, columns = len(found), len(FEATURED) + 1
    # Laid out in inches rather than by a gridspec, so that the reference
    # spans the rows exactly, its top and bottom edges on theirs, whatever its
    # aspect ratio. A magnification is square; the reference is as wide as its
    # aspect ratio makes it at that height, and a gutter sets it apart from
    # the magnifications it indexes.
    gap, gutter = 0.06, 0.3  # both in units of one magnification's side
    span = rows + (rows - 1) * gap
    aspect = width / height
    drawn = aspect if REFERENCE_ASPECT is None else min(aspect, REFERENCE_ASPECT)
    unit = WIDTH / (drawn * span + gutter + columns + (columns - 1) * gap)
    above, below = LABEL_ROOM  # inches kept for the titles and the scores
    tall = span * unit + above + below
    figure = plt.figure(figsize=(WIDTH, tall))
    figure.set_layout_engine("none")

    def place(left: float, bottom: float, wide: float, high: float):
        """An axes at a position given in inches from the lower left."""
        return figure.add_axes([left / WIDTH, bottom / tall, wide / WIDTH, high / tall])

    reference_wide = drawn * span * unit
    reference_tall = reference_wide / aspect
    plot = place(
        0, below + (span * unit - reference_tall) / 2, reference_wide, reference_tall
    )
    plot.imshow(panel["truth"])
    plot.set_axis_off()
    for (left, top), color in zip(found, COLORS):
        plot.add_patch(
            Rectangle((left, top), side, side, fill=False, color=color, linewidth=0.9)
        )

    winner = scores[list(FEATURED)].idxmax()  # both plotted metrics are larger-better
    shown = [("Ground truth", "truth", "")] + [
        (model, model, UNITS[metric].format(scores[model])) for model in FEATURED
    ]
    # Titles and scores each as large as their columns allow: the widest pair
    # of neighbours keeps 8% of the column spacing between them, measured in
    # the figure's own font, and both are capped at LABEL_SIZE.
    spacing = (1 + gap) * unit * 72  # points between column centres

    def fitting(labels: list[str], weight: str = "normal") -> float:
        widths = [text_width(label, weight) for label in labels]
        pairs = [(a + b) / 2 for a, b in zip(widths, widths[1:])]
        return min(LABEL_SIZE, 0.92 * spacing / max(pairs))

    title_size = fitting([title for title, _, _ in shown])
    score_size = fitting([score for _, _, score in shown if score], "bold")
    for column, (title, name, score) in enumerate(shown):
        for row, (left, top) in enumerate(found):
            plot = place(
                (drawn * span + gutter + column * (1 + gap)) * unit,
                below + (rows - 1 - row) * (1 + gap) * unit,
                unit,
                unit,
            )
            # Nearest keeps a NeRF render's few pixels crisp rather than blurred.
            plot.imshow(
                panel[name][top : top + side, left : left + side],
                interpolation="nearest",
            )
            plot.set(xticks=[], yticks=[])
            for spine in plot.spines.values():
                spine.set(visible=True, color=COLORS[row], linewidth=0.9)
            if row == 0:
                plot.set_title(
                    title,
                    fontsize=title_size,
                    pad=3,
                    color=PALETTE[0] if title.startswith("FUTON") else "black",
                )
            if row == rows - 1:  # the best score is set bold, as in the tables
                plot.set_xlabel(
                    score,
                    fontsize=score_size,
                    labelpad=3,
                    fontweight="bold" if name == winner else "normal",
                )
    return figure


def study(name: str) -> list[dict]:
    """The study's runs whose setup matches its config."""
    log_dir = ROOT / "logs" / "ablation-futon" / name
    if not log_dir.exists():
        return []
    config = ROOT / "configs" / "ablation-futon" / f"{name}.yaml"
    setups = yaml.safe_load(config.read_text())["models"]
    return [
        run
        for run in read(log_dir, "occupancy")
        if setups.get(run["model"]) == run["setup"]
    ]


def summarize(runs: list[dict]) -> pd.DataFrame:
    """Each model's mean size, time and IoU over the shapes, with its error."""
    final = results(runs)
    summary = final.groupby("model")[["parameters", "time", "iou"]].mean()
    return summary.assign(error=paired_error(final, ("iou",))[("mean", "iou")])


def basis_table(runs: list[dict]) -> dict[str, str]:
    """Every basis at the benchmark's size."""
    summary = summarize(runs)
    summary.index = [name.removeprefix("FUTON-").title() for name in summary.index]
    spec = {
        "parameters": PARAMS,
        "time": ("", "Train time (s)", False, 1),
        "iou": ("", "IoU (%)", True, 2),
    }
    errors = summary[["error"]].rename(columns={"error": "iou"})
    return table(summary, spec, errors, order=BASES, index="Basis")


def component_table(summaries: dict[str, pd.DataFrame]) -> dict[str, str]:
    """The combiner and the decoder, for both of the benchmark's bases.

    The studies share the benchmark's model and define it alike, so its row
    comes from the first that ran it. A variant nothing ran is left out.
    """
    spec = {"parameters": PARAMS}
    for basis in PAIR:
        spec[f"time_{basis}"] = (f"FUTON-{basis}", "Train time (s)", False, 1)
        spec[f"iou_{basis}"] = (f"FUTON-{basis}", "IoU (%)", True, 2)

    values, errors = {}, {}
    for label, template in VARIANTS.items():
        rows = {}
        for basis in PAIR:
            name = template.format(basis=basis)
            found = (s for s in summaries.values() if name in s.index)
            if (summary := next(found, None)) is not None:
                rows[basis] = summary.loc[name]
        if len(rows) < len(PAIR):
            continue
        values[label] = {"parameters": rows[PAIR[0]]["parameters"]} | {
            f"{field}_{basis}": rows[basis][field]
            for basis in PAIR
            for field in ("time", "iou")
        }
        errors[label] = {f"iou_{basis}": rows[basis]["error"] for basis in PAIR}
    if not values:
        return {}
    return table(
        pd.DataFrame(values).T,
        spec,
        pd.DataFrame(errors).T,
        order=list(VARIANTS),
        index="Combiner, decoder",
    )


def grid_figures(runs: list[dict], out_dir: Path) -> None:
    """IoU against the rank at each K, and against K at each rank, per basis."""
    data = results(runs).assign(iteration=0)  # one evaluation, the last
    data = data.join(data["model"].str.extract(GRID))
    data[["components", "rank"]] = data[["components", "rank"]].astype(int)
    data["iou"] = within_signal(data, "iou")
    labels = {"rank": "Rank $R$", "components": "Components $K$"}
    for basis, rows in data.groupby("basis"):
        for axis, hue in (("rank", "components"), ("components", "rank")):
            figure, plot = plt.subplots(figsize=SIZE)
            sns.lineplot(
                data=rows,
                x=axis,
                y="iou",
                hue=hue,
                style=hue,
                markers=True,
                dashes=False,
                errorbar=("se", 1),
                palette=list(RAMP[: rows[hue].nunique()]),
                markersize=4,
                markeredgecolor="white",
                markeredgewidth=0.5,
                ax=plot,
            )
            plot.set(xscale="log", xlabel=labels[axis], ylabel="IoU (%)")
            # The levels double, so tick them exactly, without log minor ticks.
            levels = sorted(rows[axis].unique())
            plot.set_xticks(levels, levels)
            plot.xaxis.set_minor_locator(NullLocator())
            # A white ground under the legend, since "best" still puts it over
            # a curve in a panel this small.
            plot.legend(
                title=labels[hue].split()[-1],
                framealpha=0.85,
                edgecolor="none",
                ncol=2,
                columnspacing=0.8,
                fontsize=6,
                title_fontsize=7,
            )
            save(figure, out_dir / f"{axis}_{basis}")


def study_curves(
    curve: pd.DataFrame, styles: dict[str, tuple[str, str, tuple]], path: Path
) -> None:
    """IoU against training time for a study's models, with its own legend.

    ``styles`` gives each model its label, colour and dashes, in the order the
    legend takes them. The band is one within-signal standard error, as in the
    task figures.
    """
    curve = curve[curve["model"].isin(styles)]
    curve = curve.assign(
        time=curve.groupby(["model", "iteration"])["time"].transform("mean"),
        iou=within_signal(curve, "iou"),
    )
    figure, plot = plt.subplots(figsize=(2.6, 2.1))
    sns.lineplot(
        data=curve,
        x="time",
        y="iou",
        hue="model",
        hue_order=list(styles),
        palette={model: color for model, (_, color, _) in styles.items()},
        style="model",
        dashes={model: dashes for model, (_, _, dashes) in styles.items()},
        errorbar=("se", 1),
        err_kws={"alpha": 0.15, "linewidth": 0},
        legend=False,
        ax=plot,
    )
    means = curve.groupby(["model", "iteration"], as_index=False)[
        ["iou", "time"]
    ].mean()
    low, high = y_range(means, "iou")
    low = max(low, FLOOR["occupancy"])
    shown = means[means["iou"].between(low, high)]["time"]
    plot.set(
        xscale="log",
        xlim=(shown.min() / 1.1, means["time"].max() * 1.1),
        ylim=(low, high),
        xlabel="Training time (s)",
        ylabel="IoU (%)",
    )
    log_ticks(plot.xaxis)
    handles = [
        Line2D([], [], color=color, dashes=dashes or (None, None), label=label)
        for label, color, dashes in styles.values()
    ]
    plot.legend(
        handles=handles,
        framealpha=0.85,
        edgecolor="none",
        ncol=2,
        columnspacing=0.8,
        fontsize=6,
    )
    save(figure, path)


def write(rendered: dict[str, str], path: Path) -> None:
    """Save a table's renderings as <path>.tex and <path>.md."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix, text in rendered.items():
        path.with_suffix(f".{suffix}").write_text(text)


def ablation(out_dir: Path) -> None:
    """The paper's tables and figures of the FUTON ablations."""
    studies = {name: study(name) for name in STUDIES}
    counts = ", ".join(
        f"{name} {len({run['model'] for run in runs})}"
        for name, runs in studies.items()
    )
    print(f"ablation: {counts} models")
    summaries = {name: summarize(runs) for name, runs in studies.items() if runs}
    if studies["basis"]:
        write(basis_table(studies["basis"]), out_dir / "basis")
    if studies["components_rank"]:
        grid_figures(studies["components_rank"], out_dir)
    if combined := component_table(summaries):
        write(combined, out_dir / "combiner_decoder")
    for name, styles in CURVES.items():
        if studies[name]:
            study_curves(curves(studies[name]), styles, out_dir / name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    parser.add_argument(
        "--signals", nargs="+", help="signals to draw (default: the task's examples)"
    )
    parser.add_argument(
        "--no-panels", action="store_true", help="skip the renders, which need a GPU"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="render panels that are already there"
    )
    parser.add_argument(
        "--orbit",
        type=int,
        default=0,
        metavar="FRAMES",
        help="also save an orbit GIF of each NeRF model, for a talk",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    style()
    out_dir = ROOT / "results"
    save(legend(), out_dir / "legend")
    for task in args.tasks:
        runs, final, curve = load(task)
        save(convergence(curve, task), out_dir / task / "convergence")
        save(tradeoff(final, task), out_dir / task / "tradeoff")
        save(tradeoff(final, task, "speed"), out_dir / task / "throughput")
        for suffix, text in tables(final, task).items():
            (out_dir / task / f"table.{suffix}").write_text(text)
        for signal in [] if args.no_panels else args.signals or EXAMPLES[task]:
            directory = panels(task, signal, runs, args.device, args.overwrite)
            if args.orbit and task == "nerf":
                chosen = [
                    run
                    for run in runs
                    if run["data"] == signal and run["model"] in FEATURED
                ]
                render_orbit(
                    signal, chosen, directory, args.device, args.orbit, args.overwrite
                )
            figure = qualitative(task, signal, final, directory)
            # PDF only: a PGF of raster panels writes each one out beside it.
            figure.savefig(
                out_dir / task / f"qualitative_{signal}.pdf",
                bbox_inches="tight",
                dpi=600,
            )
            plt.close(figure)
        print(f"{task}: written to {out_dir / task}")
    ablation(out_dir / "ablation")
    # For the text: who reaches a high IoU, and how soon.
    print(f"\ntime to {TARGET:g}% IoU on every shape (s):")
    print(time_to_target(load("occupancy")[2]).dropna().sort_values().to_string())


if __name__ == "__main__":
    main()
