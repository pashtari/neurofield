"""Command line and bookkeeping shared by the ``train_*.py`` benchmark scripts.

Each run directory ``<log-dir>/<signal>/<model>`` holds ``log.txt``,
``log.json``, ``checkpoint.pt``, the final reconstruction, and ``results.json``
with the model setup, parameter count, training time, final metrics, and
training history. Finished runs are skipped, so an interrupted sweep resumes
where it stopped; ``--overwrite`` reruns them instead.

Runs are keyed by model name, so give variants their own names or their own
``--log-dir``. Several configs are merged in order, and ``--set`` overrides a
single value::

    --config configs/ablation_futon.yaml --log-dir logs/ablation-futon
    --set models.FUTON-sinc.train.lr=0.05

To collect results::

    records = [json.loads(p.read_text()) for p in Path("logs").rglob("results.json")]
    table = pd.json_normalize(records)
"""

import argparse
import gc
import json
import sys
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
import yaml

import neurofield as nf

ROOT = Path(__file__).resolve().parent.parent

# fit(config, model, data_path, run_dir, device) -> record
FitFn = Callable[[dict, dict, Path, Path, torch.device], dict[str, Any]]
# prepare(model, data_path) -> model, for settings that depend on the signal
PrepareFn = Callable[[dict, Path], dict[str, Any]]


def build(model: dict[str, Any]) -> tuple[type[torch.nn.Module], dict[str, Any]]:
    """Return a model's ``neurofield`` class and constructor arguments."""
    kwargs = dict(model["kwargs"])
    if "output_activation" in kwargs:
        kwargs["output_activation"] = getattr(torch, kwargs["output_activation"])
    return getattr(nf, model["class"]), kwargs


def relative(path: Path) -> str:
    """A path inside the repository as a repository-relative one."""
    resolved = path.resolve()
    return str(
        resolved.relative_to(ROOT) if resolved.is_relative_to(ROOT) else resolved
    )


def merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge ``overlay`` into ``base``, in place."""
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = value
    return base


def assign(config: dict[str, Any], setting: str) -> None:
    """Apply one ``a.b.c=value`` override, with the value read as YAML."""
    path, separator, value = setting.partition("=")
    if not separator:
        raise KeyError(f"{setting!r} is not of the form key.path=value")
    *parents, key = path.split(".")
    node = config
    for name in parents:
        node = node[name]
    if key not in node:
        raise KeyError(f"{path!r} is not in the config")
    node[key] = yaml.safe_load(value)


def record(
    res: dict[str, Any], model: dict[str, Any], metrics: dict[str, Any]
) -> dict[str, Any]:
    """Combine a training result, model setup, and final metrics for saving."""
    return {
        **res["config"],
        "setup": model,
        "metrics": metrics,
        "history": res["history"],
    }


def run(
    task: str, data: Sequence[Path], fit: FitFn, prepare: PrepareFn | None = None
) -> None:
    """Fit every selected model to every selected signal of a task."""
    parser = argparse.ArgumentParser(
        description=f"Fit the models in configs/{task}.yaml to each signal."
    )
    parser.add_argument(
        "--data", nargs="+", type=Path, default=data, help="signal paths"
    )
    parser.add_argument("--models", nargs="+", help="model names (default: all)")
    parser.add_argument(
        "--config",
        nargs="+",
        type=Path,
        default=[ROOT / "configs" / f"{task}.yaml"],
        help="configs, deep-merged in order",
    )
    parser.add_argument(
        "--set",
        dest="settings",
        nargs="+",
        default=[],
        metavar="KEY.PATH=VALUE",
        help="config overrides, e.g. train.num_epochs=500",
    )
    parser.add_argument("--log-dir", type=Path, default=ROOT / "logs" / task)
    parser.add_argument(
        "--overwrite", action="store_true", help="rerun runs that already finished"
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    if missing := [str(path) for path in args.data if not path.exists()]:
        parser.error(f"no such signal: {', '.join(missing)}")
    if not args.data:
        parser.error(f"no signals found; pass --data (see scripts/train_{task}.py)")

    config: dict[str, Any] = {}
    for path in args.config:
        merge(config, yaml.safe_load(path.read_text()))
    for setting in args.settings:
        try:
            assign(config, setting)
        except KeyError as error:
            parser.error(str(error))

    models = args.models or list(config["models"])
    if unknown := [name for name in models if name not in config["models"]]:
        parser.error(f"unknown models {unknown}; choose from {list(config['models'])}")
    print(f"{task}: {len(args.data)} signals x {len(models)} models")

    failed = []
    for path in args.data:
        for name in models:
            model = config["models"][name]
            if prepare is not None:
                model = prepare(model, path)

            out = args.log_dir / path.stem / name
            results = out / "results.json"
            if results.exists() and not args.overwrite:
                if json.loads(results.read_text()).get("setup") != model:
                    print(
                        f"{out} holds a run of a different setup; keeping it. "
                        "Use --overwrite, another --log-dir, or another name.",
                        file=sys.stderr,
                    )
                continue
            print(f"\n=== {path.stem} / {name} ===")
            out.mkdir(parents=True, exist_ok=True)
            (out / "log.json").unlink(missing_ok=True)  # JSON logs append

            torch.manual_seed(config["train"]["seed"])
            try:
                result = fit(config, model, path, out, torch.device(args.device))
            except Exception:
                (out / "error.txt").write_text(traceback.format_exc())
                print(traceback.format_exc())
                failed.append(f"{path.stem}/{name}")
                continue
            finally:
                gc.collect()
                torch.cuda.empty_cache()

            result = {
                **result,  # its "model" is the class name; the config name wins
                "task": task,
                "data": path.stem,
                "model": name,
                "train_time": result["history"][-1]["elapsed"],
                # Relative to the repository, so a report reads them anywhere.
                "config": [relative(path) for path in args.config] + args.settings,
            }
            text = json.dumps(result, indent=2, default=str)
            (out / "results.json").write_text(text + "\n")
            (out / "error.txt").unlink(missing_ok=True)  # from an earlier attempt

    if failed:
        raise SystemExit(f"Failed runs (see error.txt): {', '.join(failed)}")
