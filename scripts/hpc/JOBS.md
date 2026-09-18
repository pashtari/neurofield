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

`configs/ablation_futon.yaml` holds 63 models on the occupancy task, all with
the benchmark's settings (256^3 grid, 2000 epochs, lr 1e-2), so only the model
changes. The reference is K=256 components, CP rank 144 and a one-layer MLP
decoder (131,617 parameters). 63 models x 5 shapes = 315 runs, about two hours:

```bash
scripts/hpc/train.sh --clusters=accelgor --time=4:00:00 occupancy \
  --config configs/ablation_futon.yaml --log-dir logs/ablation-futon
```

| Study | Models | What varies |
| --- | --- | --- |
| Basis | `FUTON-<basis>` | cosine, lanczos, sinc, triangle, chebyshev and legendre at the reference size |
| Components and rank | `FUTON-<basis>-K<K>-R<R>`, 48 models | K and R over {64, 128, 256, 512}, for cosine, lanczos and sinc |
| Tensor network | `FUTON-<basis>-TR-K256-R<r>`, 9 models | a tensor-ring combiner of rank 8, 12 or 16 against CP at K=256 |

A tensor ring of rank `r` has exactly the size of a CP combiner of rank `r^2`,
so the tensor-network study pairs TR `r` = 8, 12, 16 with CP `R` = 64, 144, 256.

## Tuning

`configs/tuning_futon.yaml` holds 35 FUTON variants of the benchmark's size
(about 131,600 parameters each), so a gain is a better use of the same budget:
K against R, R = K, three decoders, the learning rate, basis normalization, a
combiner bias, and the other bases. 35 models x 5 shapes = 175 runs:

```bash
scripts/hpc/train.sh --clusters=accelgor --time=3:00:00 occupancy \
  --config configs/tuning_futon.yaml --log-dir logs/tuning-futon
```

To retune one baseline, override a value and send the runs elsewhere:

```bash
scripts/hpc/train.sh --clusters=accelgor --time=2:00:00 nerf \
  --models SIREN FINER --data data/nerf/blender/lego \
  --set models.SIREN.train.lr=2.0e-4 --log-dir logs/tune-nerf/lr2e-4
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
