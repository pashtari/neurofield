# Reproducing the paper

Every number and figure of the paper comes from the training scripts, one that reports, and one that times. This page is the full pipeline, from an empty `data/` folder to `results/`.

## 1. Data

```bash
python scripts/download_kodak.py      # data/Kodak/, 24 images, about 30 MB
python scripts/download_meshes.py     # data/occupancy/, 5 meshes, about 3 GB
python scripts/download_blender.py    # data/nerf/blender/, 8 scenes, about 2.4 GB
python scripts/download_div2k.py      # data/DIV2K/, 100 validation images and their 4x downsamplings, about 450 MB
```

The Kodak images come from Rich Franzen's mirror, the meshes from the Stanford 3D Scanning Repository (rotated into canonical pose where a release is not), the Blender scenes from a pinned Hugging Face mirror of the official Google Drive release, and the DIV2K images from the official site. All four scripts resume and keep existing files.

## 2. Training

`scripts/train_<task>.py` fits every model of `configs/<task>.yaml` to every signal of the task:

```bash
python scripts/train_image.py        # 24 images x 12 models -> logs/image/<image>/<model>/
python scripts/train_occupancy.py    # 5 shapes x 13 models  -> logs/occupancy/<shape>/<model>/
python scripts/train_nerf.py         # 8 scenes x 13 models  -> logs/nerf/<scene>/<model>/
python scripts/train_super_resolution.py  # 100 images x 16 models -> logs/super_resolution/<image>/<model>/
```

The four scripts share one command line:

| Option | Meaning |
| --- | --- |
| `--data PATH ...` | Signals to fit; default: every signal under `data/` for the task. |
| `--models NAME ...` | Model names from the config; default: all. |
| `--config FILE ...` | Configs, deep-merged in order; the last names the log directory (`configs/<name>.yaml` writes to `logs/<name>/`). |
| `--set KEY.PATH=VALUE ...` | Override one value, read as YAML, e.g. `train.num_epochs=500` or `models.SIREN.train.lr=3e-4`. |
| `--log-dir DIR` | Where the runs go. |
| `--overwrite` | Rerun runs that already finished. |
| `--device DEV` | `cuda` when available, else `cpu`. |

A config holds the shared `data` and `train` settings and, per model, its NeuroField class, constructor `kwargs` and training overrides (the learning rate, and for the hash grid its Adam settings). Image configs may use expressions in the image height `H` and width `W`, such as `[H // 2, W // 2]`; the super-resolution config sizes its models from the high-resolution image, and FUTON's entry names a parameter `budget` whose largest CP rank the script fills in. Comments beside each learning rate record what the neighbouring values cost, and `# deviation` marks a setting that departs from the authors' code and why.

Finished runs are skipped, so an interrupted sweep restarts where it stopped, and a failed run writes `error.txt` and lets the sweep continue. Runs are keyed by model name, so a variant needs its own name or its own `--log-dir`:

```bash
python scripts/train_image.py --models Gauss --overwrite                       # rerun one model in place
python scripts/train_image.py --set models.SIREN.train.lr=0.001 --log-dir logs/siren-lr
python scripts/train_image.py --config configs/ablation-futon/basis.yaml       # -> logs/ablation-futon/basis/
```

Each run directory holds `log.txt`, `log.json`, `checkpoint.pt`, the final reconstruction (images) or two rendered test views (NeRF; super-resolution saves none, the report renders them), and `results.json` with the training configuration, the model's `setup` from the YAML, `num_params`, `train_time`, the final `metrics` (NeRF: the mean over the 200 test views, with every view under `test_views`) and the `history`. Occupancy runs write no mesh, since rendering needs a display; the report rebuilds them from the checkpoints.

Measured on one A100 (2000 epochs; NeRF 37,500 steps):

| Task | Per model | Per signal | Whole sweep |
| --- | --- | --- | --- |
| Images | 37 s | 7.5 min | 3 h |
| Occupancy | 19 s | 4 min | 20 min |
| NeRF | 10 min | 2.2 h | 18 h |
| Super-resolution | 6.5 min | 85 min | 6 days |

The image figure includes the 20 LPIPS evaluations of a run; training alone takes about 17 s. The super-resolution times are from an RTX 4060 Ti, which this workload runs at about the A100's pace: each step renders 261k pixels through a 500k-parameter model, so the sweep is best spread over many short jobs, one image each. Every run uses seed 0; the FUTON kernels have a deterministic backward pass, so a rerun on the same hardware and PyTorch build reproduces a run's numbers, while training times vary with the machine and its load.

## 3. Ablations

The FUTON ablations are configs in `configs/ablation-futon/`, run through the same scripts. All keep the benchmark's settings, so only the model changes:

| Config | Task | Models | Varies |
| --- | --- | --- | --- |
| `basis.yaml` | image | 6 | The six bases at the benchmark's size. |
| `components_rank.yaml` | image | 50 | The components per axis as a fraction $\alpha$ of the pixels, $K = (\alpha H, \alpha W)$, and the rank as a fraction $\beta$ of the smaller count, $R = \beta \min(K_1, K_2)$, both over {1/8, 1/4, 1/2, 1, 2}, sinc and Lanczos. |
| `tensor_net.yaml` | image | 4 | Tensor-ring against CP, both bases. |
| `decoder.yaml` | image | 4 | Linear against MLP decoder, at equal size. |
| `components_nerf.yaml` | nerf | 2 | $K = 256$ at the rank that pays for it, against the benchmark. |

The image studies run on all 24 Kodak images at the benchmark's settings (2000 epochs on 10 % of the pixels per step, learning rate 3e-2); a model's name carries its fractions as decimals, so `FUTON-sinc-a0.5-b1` has $K = (H/2, W/2)$ and $R = H/2$.

```bash
python scripts/train_image.py --config configs/ablation-futon/components_rank.yaml
python scripts/train_nerf.py --config configs/ablation-futon/components_nerf.yaml --data data/nerf/blender/lego
```

## 4. Report

`scripts/report_paper.py` turns the runs into `results/`, which holds the paper's figures and tables and nothing else:

```
results/
├── legend.{pdf,pgf}                 the models' legend, shared by every figure; super_resolution/legend adds the bicubic level
├── <task>/convergence.{pdf,pgf}     quality against training time, FUTON and the strongest model of each family
├── <task>/training_time.{pdf,pgf}   every model's final quality against its training time
├── <task>/throughput.{pdf,pgf}      the same against its inference throughput
├── <task>/{training,inference}_memory.{pdf,pgf}  and against its peak memory in either phase
├── <task>/table.{tex,md}            size, the cost of both phases and final metrics, every shape's and scene's too, best in bold, second underlined
├── <task>/qualitative_<signal>.pdf  the signal with two regions boxed and magnified for every featured model (bicubic for TensoRF in super-resolution)
├── <task>/panels/<signal>/          the renders those magnifications are cut from, rebuilt from the checkpoints
├── ablation/                        a table and a convergence plot each for the bases, the combiners and the decoders, and PSNR against each fraction
└── bound/                           the error bound against trained FUTONs, from the records of scripts/futon_bound.py: figures and tables of the bound page
```

```bash
python scripts/report_paper.py                          # everything
python scripts/report_paper.py --tasks image --overwrite  # redraw one task's panels
python scripts/report_paper.py --tasks bound            # only the error bound; --tasks also takes ablation
python scripts/report_paper.py --no-panels              # tables and plots only, no GPU needed
python scripts/report_paper.py --orbit 60               # plus an orbit GIF of each NeRF model
```

The magnified regions are found, not chosen: the two squares of fine detail where FUTON gains most squared error over the strongest baseline. Drawing the panels needs the signals in `data/` and a GPU; they are kept and reused, so recomposing a figure costs seconds. Figures come as PDF and as PGF for LaTeX, sized for a NeurIPS page: a plot is a third of the 5.5-inch text width, a strip of magnifications spans it, text runs from 6.5 to 8 pt, and every tick falls on a whole or half number. The image and super-resolution tables fit the text width in 9 pt type; the occupancy and NeRF ones, with every shape's and scene's scores, are wider and take a page turned sideways. The `±` in the tables is a paired standard error over the signals: each signal's own level is removed before the error is taken, which is the uncertainty of a comparison between models rather than of an absolute level; IoU, which saturates near 100 % on every shape, takes the plain standard error over the shapes. A band in a figure spans one of them, the same error.

Timings recorded during a sweep carry whatever else shared the node: retrained alone on an exclusive node, Instant-NGP was 1.7× faster, MFN and TensoRF-VM 1.2×, the other models within a few percent. The report therefore takes accuracy from the sweeps and times from separate exclusive jobs, the recipe in `scripts/hpc/JOBS.md`: `scripts/profile_speed.py` times every model's inference on one signal per task and writes `logs/speed/<task>.json`, which the throughput figure reads, and the same training scripts rerun every model into `logs/timing/<task>/`, after a warm-up pass that keeps kernel compilation out of the clock. Wherever a timing run exists, its times replace the sweep's in the tables and figures while the metrics stay the sweep's (they agree where training is deterministic), and the mean training time is taken over the timed signals: every image and shape, the lego and hotdog scenes, and DIV2K images 0801 and 0882.

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
scripts/hpc/train.sh --clusters=accelgor --time=2:00:00 --array=801-900 super_resolution \
  --data data/DIV2K/DIV2K_valid_HR/%4a.png   # one image per array task
scripts/hpc/pull_logs.sh
python scripts/report_paper.py
```
