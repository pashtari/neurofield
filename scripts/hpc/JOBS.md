# Jobs

Every run in this work, as submitted on HPC-UGent. See `README.md` for the
setup and for how `train.sh` works. All jobs use one A100 (accelgor), one job
at a time per line below; finished runs are skipped, so a command that stops
early can simply be repeated.

## Benchmarks

The 13 (image) or 14 (3D) models of `configs/<task>.yaml` on every signal.

| Job | Runs | Time | Logs |
| --- | --- | --- | --- |
| image | 24 images x 13 models = 312 | ~3.5 h | `logs/image/<image>/<model>/` |
| occupancy | 5 shapes x 14 models = 70 | ~25 min | `logs/occupancy/<shape>/<model>/` |
| nerf | 8 scenes x 14 models = 112 | ~2.5 h per scene | `logs/nerf/<scene>/<model>/` |

```bash
scripts/hpc/train.sh --clusters=accelgor --time=5:00:00 image
scripts/hpc/train.sh --clusters=accelgor --time=1:00:00 occupancy
for scene in data/nerf/blender/*/; do
  scripts/hpc/train.sh --clusters=accelgor --time=4:00:00 nerf --data "$scene"
done
```

## FUTON ablations

`configs/ablation_futon.yaml` holds 41 models on the occupancy task, which is
the cheapest of the three and the most sensitive to the basis. They all share
the benchmark's settings (256^3 grid, 2000 epochs, lr 1e-2), so only the model
changes. 41 models x 5 shapes = 205 runs, about 1.5 hours in total:

```bash
scripts/hpc/train.sh --clusters=accelgor --time=3:00:00 occupancy \
  --config configs/ablation_futon.yaml --log-dir logs/ablation-futon
```

The runs land in `logs/ablation-futon/<shape>/<model>/`, separate from the
benchmark, and cover three questions:

| Study | Models | What varies |
| --- | --- | --- |
| Reference | `FUTON-cosine`, `FUTON-sinc` | the benchmark entries, repeated (K=256, R=144, 131,617 parameters) |
| Rank and components | `FUTON-<basis>-K<K>-R<R>`, 32 models | K and R over {64, 128, 256, 512} for both bases (16,513 to 1,049,601 parameters) |
| Tensor ring | `FUTON-<basis>-TR-K256-R<r>`, 4 models | a TR combiner in place of CP, r = 12 and 16 |
| Basis | `FUTON-triangle`, `FUTON-chebyshev`, `FUTON-legendre` | three more bases at the reference size |

A TR combiner of rank `r` has exactly the parameter count of a CP combiner of
rank `r^2`, so `TR-K256-R12` pairs with the reference `R=144` and
`TR-K256-R16` with `K256-R256`, each at equal size.

To run one study alone, add `--models`, e.g.
`--models FUTON-chebyshev FUTON-legendre`. To split the sweep over several
GPUs, give each job its own shapes with `--data`.

## Tuning

`configs/tuning_futon.yaml` holds 22 FUTON variants of the benchmark's size
(about 131,600 parameters each), so a gain is a better use of the same budget:
components against rank at fixed size, three decoders, two learning rates, the
basis normalization, a combiner bias, and the other bases. 22 models x 5
shapes = 110 runs, about an hour:

```bash
scripts/hpc/train.sh --clusters=accelgor --time=3:00:00 occupancy \
  --config configs/tuning_futon.yaml --log-dir logs/tuning-futon
```

To retune one baseline, override its rate and send the runs elsewhere:

```bash
scripts/hpc/train.sh --clusters=accelgor --time=2:00:00 nerf \
  --models SIREN FINER --data data/nerf/blender/lego \
  --set models.SIREN.train.lr=3.0e-3 --log-dir logs/tune-nerf/lr3e-3
```

A config change to a model that has already run needs `--overwrite`, which
reruns it in place:

```bash
scripts/hpc/train.sh --clusters=accelgor --time=2:00:00 image \
  --models Gauss --overwrite
```

## Reports

Training keeps every run's `checkpoint.pt` but writes no figures, so tables,
plots and examples are made afterwards, from the pulled logs:

```bash
scripts/hpc/pull_logs.sh                # on the local machine
python scripts/report_image.py          # -> results/image/
python scripts/report_occupancy.py      # -> results/occupancy/
python scripts/report_nerf.py           # -> results/nerf/
python scripts/report_ablation.py       # -> results/ablation-futon/
```

Each writes one table (CSV, Markdown, LaTeX) of the models: their size,
times and metrics averaged over the signals as `mean±std`, then one super
column per signal (image, with 24 of them, keeps the average alone),
convergence plots per metric against iteration and time as PDF and PGF, and
qualitative examples rendered from the checkpoints, each signal's also
composed into one `comparison.pdf`; `--qualitative` picks the signals, `--qualitative none` skips
them. Rendering a mesh needs a display, so run the occupancy report on a
workstation. NeRF renders need none and can run on the cluster:

```bash
sbatch --clusters=accelgor --time=1:00:00 --gpus-per-node=1 --cpus-per-task=8 \
  --chdir=$VSC_DATA/projects/neurofield --job-name=nf-report --wrap \
  "source $VSC_DATA/venvs/neurofield-env/bin/activate && python scripts/report_nerf.py"
```

## Collecting the results

```bash
scripts/hpc/pull_logs.sh   # from the local machine
```

```python
records = [json.loads(p.read_text()) for p in Path("logs").rglob("results.json")]
table = pd.json_normalize(records)  # task, data, model, num_params, metrics.*, setup.*
```

`setup.kwargs.combiner` and `setup.kwargs.basis` carry the rank, the number of
components and the basis of each run, so the ablation tables group directly by
them.
