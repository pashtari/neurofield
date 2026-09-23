# Reproducing the paper

Every number and figure of the paper comes from three scripts: one that trains, one that reports, and one that times. This page is the full pipeline, from an empty `data/` folder to `results/`.

## 1. Data

```bash
python scripts/download_kodak.py      # data/Kodak/, 24 images, about 30 MB
python scripts/download_meshes.py     # data/occupancy/, 5 meshes, about 3 GB
python scripts/download_blender.py    # data/nerf/blender/, 8 scenes, about 2.4 GB
```

The Kodak images come from Rich Franzen's mirror, the meshes from the Stanford 3D Scanning Repository (rotated into canonical pose where a release is not), and the Blender scenes from a pinned Hugging Face mirror of the official Google Drive release. All three scripts resume and keep existing files.

## 2. Training

`scripts/train_<task>.py` fits every model of `configs/<task>.yaml` to every signal of the task:

```bash
python scripts/train_image.py        # 24 images x 12 models -> logs/image/<image>/<model>/
python scripts/train_occupancy.py    # 5 shapes x 13 models  -> logs/occupancy/<shape>/<model>/
python scripts/train_nerf.py         # 8 scenes x 13 models  -> logs/nerf/<scene>/<model>/
```

The three scripts share one command line:

| Option | Meaning |
| --- | --- |
| `--data PATH ...` | Signals to fit; default: every signal under `data/` for the task. |
| `--models NAME ...` | Model names from the config; default: all. |
| `--config FILE ...` | Configs, deep-merged in order; the last names the log directory (`configs/<name>.yaml` writes to `logs/<name>/`). |
| `--set KEY.PATH=VALUE ...` | Override one value, read as YAML, e.g. `train.num_epochs=500` or `models.SIREN.train.lr=3e-4`. |
| `--log-dir DIR` | Where the runs go. |
| `--overwrite` | Rerun runs that already finished. |
| `--device DEV` | `cuda` when available, else `cpu`. |

A config holds the shared `data` and `train` settings and, per model, its NeuroField class, constructor `kwargs` and training overrides (the learning rate, and for the hash grid its Adam settings). Image configs may use expressions in the image height `H` and width `W`, such as `[H // 2, W // 2]`. Comments beside each learning rate record what the neighbouring values cost, and `# deviation` marks a setting that departs from the authors' code and why.

Finished runs are skipped, so an interrupted sweep restarts where it stopped, and a failed run writes `error.txt` and lets the sweep continue. Runs are keyed by model name, so a variant needs its own name or its own `--log-dir`:

```bash
python scripts/train_image.py --models Gauss --overwrite                       # rerun one model in place
python scripts/train_image.py --set models.SIREN.train.lr=0.001 --log-dir logs/siren-lr
python scripts/train_occupancy.py --config configs/ablation-futon/basis.yaml   # -> logs/ablation-futon/basis/
```

Each run directory holds `log.txt`, `log.json`, `checkpoint.pt`, the final reconstruction (images) or two rendered test views (NeRF), and `results.json` with the training configuration, the model's `setup` from the YAML, `num_params`, `train_time`, the final `metrics` (NeRF: the mean over the 200 test views, with every view under `test_views`) and the `history`. Occupancy runs write no mesh, since rendering needs a display; the report rebuilds them from the checkpoints.

Measured on one A100 (2000 epochs; NeRF 37,500 steps):

| Task | Per model | Per signal | Whole sweep |
| --- | --- | --- | --- |
| Images | 37 s | 7.5 min | 3 h |
| Occupancy | 19 s | 4 min | 20 min |
| NeRF | 10 min | 2.2 h | 18 h |

The image figure includes the 20 LPIPS evaluations of a run; training alone takes about 17 s. Every run uses seed 0; the FUTON kernels have a deterministic backward pass, so a rerun on the same hardware and PyTorch build reproduces a run's numbers, while training times vary with the machine and its load.

## 3. Ablations

The FUTON ablations are configs in `configs/ablation-futon/`, run through the same scripts. All keep the benchmark's settings, so only the model changes:

| Config | Task | Models | Varies |
| --- | --- | --- | --- |
| `basis.yaml` | occupancy | 6 | The six bases at the default size. |
| `components_rank.yaml` | occupancy | 32 | $K$ and $R$ over {32, 64, 128, 256}, sinc and Lanczos. |
| `tensor_net.yaml` | occupancy | 4 | Tensor-ring against CP, both bases. |
| `decoder.yaml` | occupancy | 6 | Linear against MLP decoder, at equal size and at equal rank. |
| `decoder_image.yaml` | image | 12 | The linear decoder along the iso-parameter curve, with MLP controls. |
| `components_nerf.yaml` | nerf | 2 | $K = 256$ at the rank that pays for it, against the benchmark. |

```bash
python scripts/train_occupancy.py --config configs/ablation-futon/components_rank.yaml
python scripts/train_image.py --config configs/ablation-futon/decoder_image.yaml --data data/Kodak/kodim19.png
python scripts/train_nerf.py --config configs/ablation-futon/components_nerf.yaml --data data/nerf/blender/lego
```

## 4. Report

`scripts/report_paper.py` turns the runs into `results/`, which holds the paper's figures and tables and nothing else:

```
results/
├── legend.{pdf,pgf}                 the models' legend, shared by every figure
├── <task>/convergence.{pdf,pgf}     quality against training time, FUTON and the strongest model of each family
├── <task>/tradeoff.{pdf,pgf}        every model's final quality against its training time
├── <task>/throughput.{pdf,pgf}      the same against its inference rate
├── <task>/table.{tex,md}            size, training time and final metrics, best in bold, second underlined
├── <task>/qualitative_<signal>.pdf  the signal with two regions boxed and magnified for every featured model
├── <task>/panels/<signal>/          the renders those magnifications are cut from, rebuilt from the checkpoints
└── ablation/                        the basis table and plot, the combiner table and plot, IoU against rank and against K
```

```bash
python scripts/report_paper.py                          # everything
python scripts/report_paper.py --tasks image --overwrite  # redraw one task's panels
python scripts/report_paper.py --no-panels              # tables and plots only, no GPU needed
python scripts/report_paper.py --orbit 60               # plus an orbit GIF of each NeRF model
```

The magnified regions are found, not chosen: the two squares of fine detail where FUTON gains most squared error over the strongest baseline. Drawing the panels needs the signals in `data/` and a GPU; they are kept and reused, so recomposing a figure costs seconds. Figures come as PDF and as PGF for LaTeX. The `±` in the tables is a paired standard error over the signals: each signal's own level is removed before the error is taken, which is the uncertainty of a comparison between models rather than of an absolute level.

Timings recorded during a sweep carry whatever else shared the node: retrained alone on an exclusive node, Instant-NGP was 1.7× faster, MFN and TensoRF-VM 1.2×, the other models within a few percent. The report therefore takes accuracy from the sweeps and times from separate exclusive jobs, the recipe in `scripts/hpc/JOBS.md`: `scripts/profile_speed.py` times every model's inference on one signal per task and writes `logs/speed/<task>.json`, which the throughput figure reads, and the same training scripts rerun every model into `logs/timing/<task>/`, after a warm-up pass that keeps kernel compilation out of the clock. Wherever a timing run exists it replaces the sweep's run in the tables and figures (training is deterministic, so the metrics agree), and the mean training time is taken over the timed signals.

Or skip the report and read the runs directly:

```python
import json
from pathlib import Path
import pandas as pd

records = [json.loads(p.read_text()) for p in Path("logs").rglob("results.json")]
table = pd.json_normalize(records)
```

## 5. On a cluster

`scripts/hpc/` runs the sweeps as Slurm jobs on the HPC-UGent clusters: `setup.sh` prepares a clone, a virtual environment and the data once; `train.sh` submits one task (or one signal, or one model) as a GPU job, pushing and checking out the current commit first; `pull_logs.sh` copies `logs/` back. `scripts/hpc/README.md` documents them and `scripts/hpc/JOBS.md` lists every job of this work. The scripts are specific to that site's paths and SSH host, but `train.sh` is a short template for any Slurm cluster.

```bash
scripts/hpc/train.sh --clusters=accelgor --time=5:00:00 image
for scene in data/nerf/blender/*/; do
  scripts/hpc/train.sh --clusters=accelgor --time=4:00:00 nerf --data "$scene"
done
scripts/hpc/pull_logs.sh
python scripts/report_paper.py
```
