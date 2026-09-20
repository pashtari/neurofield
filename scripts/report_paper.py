"""The paper's figures and tables, one of each per task for LaTeX subfigures.

Usage:
    python scripts/report_paper.py

Writes results/paper/:
    legend.{pdf,pgf}               the models' legend, shared by every figure
    <task>/convergence.{pdf,pgf}   quality against training time, on log axes, for
                                   FUTON and the strongest model of each other family
    <task>/tradeoff.{pdf,pgf}      every model's final quality against its training time
    <task>/throughput.{pdf,pgf}    the same against its inference rate
    <task>/table.{tex,md}          every model's size, training time and final metrics,
                                   one row each, the averages with a paired standard
                                   error: for occupancy every shape's IoU, and for NeRF
                                   every scene's metrics under a super column

Tables mark the best value in bold and the second underlined. The per-task
reports hold every model's curves and the tables with standard deviations.
"""

from collections.abc import Sequence
from itertools import groupby

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import report_common as report
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.ticker import (
    FixedLocator,
    FuncFormatter,
    NullFormatter,
    StrMethodFormatter,
)

# FUTON in one hue, sinc solid and lanczos dashed, and the strongest model of
# each other family: periodic activations (FINER), hash grids (Instant-NGP)
# and factorized grids (TensoRF). Color, dashes and whether the marker is filled.
FEATURED = {
    "FUTON-sinc": (report.PALETTE[0], "", True),
    "FUTON-lanczos": (report.PALETTE[0], (4, 2), False),
    "FINER": (report.PALETTE[1], "", True),
    "Instant-NGP": (report.PALETTE[2], "", True),
    "TensoRF": (report.PALETTE[6], "", True),
}
OTHERS = "0.7"  # gray of every other model
# TensoRF is its default variant for the dimension: CP in 2D, where it is the
# only one, and VM in 3D, the one its authors recommend. The per-task reports
# keep both.
TENSORF = {"image": "TensoRF", "occupancy": "TensoRF-VM", "nerf": "TensoRF-VM"}
# Per task, the metric plotted, its axis label and scale, and the metrics
# tabulated. IoU is drawn on a logit axis, which expands the saturated end,
# where 99.7% and 99.9% differ threefold in error, and keeps larger upwards.
TASKS = {
    "image": ("psnr", "PSNR (dB)", "linear", ("psnr", "ssim", "lpips")),
    "occupancy": ("fraction", "IoU (%)", "logit", ("iou",)),
    "nerf": ("psnr", "PSNR (dB)", "linear", ("psnr", "ssim", "lpips")),
}
ROWS = (
    "RFF", "PE-MLP", "MFN", "SIREN", "Gauss", "WIRE", "FINER", "Instant-NGP",
    "TensoRF", "GA-Planes", "FUTON-sinc", "FUTON-lanczos",
)  # fmt: skip
TARGET = 99.8  # IoU (%) whose first time the occupancy table reports
LOGIT_TICKS = (0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995, 0.9999)
SIZE = (2.3, 1.9)  # inches, a third of a two-column page with gutters


def load(task: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A task's final metrics and its curves, one row per run and per evaluation."""
    runs = report.read(report.ROOT / "logs" / task, task)
    frames = []
    for frame in (report.results(runs), report.curves(runs)):
        # One TensoRF, the task's default variant.
        others = {"TensoRF", "TensoRF-CP", "TensoRF-VM"} - {TENSORF[task]}
        frame = frame[~frame["model"].isin(others)]
        frame = frame.replace({"model": {TENSORF[task]: "TensoRF"}})
        if "iou" in frame:
            frame["fraction"] = frame["iou"] / 100
        frames.append(frame)
    return tuple(frames)


def cost() -> dict:
    """The size and training time every table opens with, as TensoRF, K-Planes
    and NeuRBF report them; the inference rate stays in the per-task reports."""
    return {
        "parameters": ("", "# Params (k)", None, 1),
        "time": ("", "Train time (s)", False, 1),
    }


def featured(frame: pd.DataFrame) -> pd.DataFrame:
    """The rows of the models the figures show."""
    return frame[frame["model"].isin(FEATURED)]


def nice_ticks(plot: Axes) -> None:
    """Plain labels on log and logit axes: 1, 2, 5, 10 ... and 99, 99.9 (%).

    A log axis takes the densest of the 1-2-3-5, 1-2-5 and 1 ticks per decade
    that keeps to five within its limits.
    """
    for axis, (low, high) in (
        (plot.xaxis, plot.get_xlim()),
        (plot.yaxis, plot.get_ylim()),
    ):
        if axis.get_scale() == "logit":
            ticks = [tick for tick in LOGIT_TICKS if low <= tick <= high]
            axis.set_major_formatter(FuncFormatter(lambda value, _: f"{100 * value:g}"))
        elif axis.get_scale() == "log":
            decades = range(int(np.log10(low)) - 1, int(np.log10(high)) + 2)
            for subs in ((1, 2, 3, 5), (1, 2, 5), (1,)):
                ticks = [
                    m * 10.0**k
                    for k in decades
                    for m in subs
                    if low <= m * 10.0**k <= high
                ]
                if len(ticks) <= 5:
                    break
            axis.set_major_formatter(StrMethodFormatter("{x:g}"))
        else:
            continue
        axis.set_major_locator(FixedLocator(ticks))
        axis.set_minor_formatter(NullFormatter())


def limits(means: pd.DataFrame, metric: str, scale: str) -> tuple[float, float]:
    """A y range on which the models separate, found in the axis' own scale."""
    forward, inverse = {
        "log": (np.log10, lambda v: 10**v),
        "logit": (lambda p: np.log(p / (1 - p)), lambda v: 1 / (1 + np.exp(-v))),
    }.get(scale, (lambda value: value,) * 2)
    found = report.y_range(
        means.assign(mean=forward(means[metric])).reset_index(), "iteration", True
    )
    return tuple(inverse(np.array(found)))


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


def convergence(curves: pd.DataFrame, task: str) -> plt.Figure:
    """The task's metric against training time, for the featured models."""
    metric, label, scale, _ = TASKS[task]
    curves = featured(curves)
    curves = curves.assign(
        time=curves.groupby(["model", "iteration"])["time"].transform("mean"),
        **{metric: report.within_signal(curves, metric)},
    )
    figure, plot = plt.subplots(figsize=SIZE)
    sns.lineplot(
        data=curves,
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
    means = curves.groupby(["model", "iteration"])[[metric, "time"]].mean()
    low, high = limits(means, metric, scale)
    shown = means[means[metric].between(low, high)]["time"]
    # Scaled after plotting, since seaborn would average in the axis' scale.
    plot.set(
        xscale="log",
        yscale=scale,
        xlim=(shown.min() / 1.1, means["time"].max() * 1.1),
        ylim=(low, high),
        xlabel="Training time (s)",
        ylabel=label,
    )
    nice_ticks(plot)
    return figure


def tradeoff(final: pd.DataFrame, task: str, cost: str = "time") -> plt.Figure:
    """Each model's final quality against its training time or inference rate."""
    metric, label, scale, _ = TASKS[task]
    axis = {
        "time": "Training time (s)",
        "speed": f"Inference ({report.SPEED[task]})",
    }[cost]
    means = featured(final).groupby("model")[[cost, metric]].mean()
    rest = final[~final["model"].isin(FEATURED)]
    others = rest.groupby("model")[[cost, metric]].mean()
    figure, plot = plt.subplots(figsize=SIZE)
    plot.set(
        xscale="log",
        yscale=scale,
        xlim=(final[cost].min() / 1.15, final[cost].max() * 1.15),
        xlabel=axis,
        ylabel=label,
    )
    plot.scatter(others[cost], others[metric], s=14, color=OTHERS, zorder=2)
    for model, (color, _, filled) in FEATURED.items():
        x, y = means.loc[model, [cost, metric]]
        face = color if filled else "white"
        plot.scatter(x, y, s=30, facecolor=face, edgecolor=color, zorder=3)
    nice_ticks(plot)
    return figure


def table(
    values: pd.DataFrame, spec: dict, errors: pd.DataFrame | None = None
) -> dict[str, str]:
    """LaTeX and Markdown of models by columns, best bold and second underlined.

    ``spec`` maps each column of ``values`` to its group (empty for none), its
    header, whether larger is better (``None`` not to rank it) and its
    decimals; values tied at those decimals share a rank. A column of
    ``errors`` follows its value as ``mean+-error``. Consecutive columns of one
    group share a spanning header, and FUTON's rows come last.
    """
    models = [model for model in ROWS if model in values.index]
    cells = {}
    for name, (_, _, larger_is_better, decimals) in spec.items():
        shown = values[name].reindex(models).round(decimals)
        error = None if errors is None or name not in errors else errors[name]
        ranks = shown.rank(ascending=not larger_is_better, method="min")
        for model in models:
            text = "–" if pd.isna(shown[model]) else f"{shown[model]:.{decimals}f}"
            if error is not None and not pd.isna(shown[model]):
                text += f"±{error[model]:.{decimals}f}"
            rank = ranks[model] if larger_is_better is not None else np.nan
            cells[model, name] = text, rank

    def line(model: str, bold: str, underline: str) -> list[str]:
        return [model] + [
            bold.format(text) if rank == 1 else underline.format(text) if rank == 2
            else text
            for text, rank in (cells[model, name] for name in spec)
        ]  # fmt: skip

    titles = ["Model"] + [
        title if larger_is_better is None
        else f"{title} {'↑' if larger_is_better else '↓'}"
        for _, title, larger_is_better, _ in spec.values()
    ]  # fmt: skip
    groups = [""] + [group for group, *_ in spec.values()]
    latex = [rf"\begin{{tabular}}{{l{'c' * (len(titles) - 1)}}}", r"\toprule"]
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
    latex += [" & ".join(titles) + r" \\", r"\midrule"]
    markdown = ["| " + " | ".join(flat) + " |", "| --- " * len(flat) + "|"]
    for model in models:
        if model == "FUTON-sinc":
            latex.append(r"\midrule")
        latex.append(
            " & ".join(line(model, r"\textbf{{{}}}", r"\underline{{{}}}")) + r" \\"
        )
        markdown.append("| " + " | ".join(line(model, "**{}**", "_{}_")) + " |")
    latex += [r"\bottomrule", r"\end{tabular}"]
    tex = "\n".join(latex) + "\n"
    # None of the commands above holds one of these, so escaping is safe.
    for symbol, command in (
        ("↑", r"$\uparrow$"),
        ("↓", r"$\downarrow$"),
        ("±", r"$\pm$"),
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
            name, larger_is_better, _, decimals = report.METRICS[metric]
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
    frame.columns = pd.Index(list(values), tupleize_cols=False)
    return frame, spec


def time_to_target(curves: pd.DataFrame) -> pd.Series:
    """Mean time each model first reaches the target IoU, if on every shape."""
    reached = curves[curves["iou"] >= TARGET].groupby(["model", "signal"])["time"]
    first = reached.min().groupby("model")
    return first.mean().where(first.size() == curves["signal"].nunique())


def paired_error(final: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    """Each model's standard error over signals, once their levels are removed.

    Signals differ far more than models do, so the plain deviation over them
    would say how hard the signals are; see :func:`report.within_signal`.
    """
    final = final.assign(iteration=0)  # one evaluation, the last
    return pd.DataFrame(
        {
            ("mean", metric): final.assign(
                **{metric: report.within_signal(final, metric)}
            )
            .groupby("model")[metric]
            .sem()
            for metric in metrics
        }
    )


def tables(final: pd.DataFrame, curves: pd.DataFrame, task: str) -> dict[str, str]:
    """The task's table, with its metrics per signal where the signals are few.

    The averaged columns carry one paired standard error over the signals.
    NeRF's table has a super column per scene over its three metrics, so it
    needs the width of a page turned sideways.
    """
    metrics = TASKS[task][3]
    spec = cost()
    values = final.groupby("model")[list(spec)].mean()
    errors = paired_error(final, metrics)
    if task == "image":  # 24 images, so their mean alone
        for metric in metrics:
            name, larger_is_better, _, decimals = report.METRICS[metric]
            spec[metric] = ("", name, larger_is_better, decimals)
        values = values.join(final.groupby("model")[list(metrics)].mean())
        errors.columns = list(metrics)
        return table(values, spec, errors)
    per, per_spec = per_signal(final, metrics)
    return table(values.join(per), spec | per_spec, errors)


def main() -> None:
    out_dir = report.ROOT / "results" / "paper"
    report.style()
    report.save(legend(), out_dir / "legend")
    for task in TASKS:
        final, curves = load(task)
        report.save(convergence(curves, task), out_dir / task / "convergence")
        report.save(tradeoff(final, task), out_dir / task / "tradeoff")
        report.save(tradeoff(final, task, "speed"), out_dir / task / "throughput")
        for suffix, text in tables(final, curves, task).items():
            (out_dir / task / f"table.{suffix}").write_text(text)
        print(f"{task}: written to {out_dir / task}")
    # For the text: who reaches a high IoU, and how soon.
    print(f"\ntime to {TARGET:g}% IoU on every shape (s):")
    print(time_to_target(load("occupancy")[1]).dropna().sort_values().to_string())


if __name__ == "__main__":
    main()
