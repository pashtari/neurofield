# Jobs

Every job of this work, all on one A100 of accelgor. `README.md` covers the
setup and how `train.sh` works. Each command below submits one job (the NeRF
loop, one per scene). Finished runs are skipped, so a job that stops early is
simply submitted again; `--overwrite` reruns finished runs instead.

## Benchmarks

Every model of `configs/<task>.yaml` on every signal of the task:

| Task | Runs | Time | Logs |
| --- | --- | --- | --- |
| image | 24 Kodak images x 13 models = 312 | ~3.5 h | `logs/image/<image>/<model>/` |
| occupancy | 5 Stanford shapes x 14 models = 70 | ~25 min | `logs/occupancy/<shape>/<model>/` |
| nerf | 8 Blender scenes x 14 models = 112 | ~2.5 h per scene | `logs/nerf/<scene>/<model>/` |

```bash
scripts/hpc/train.sh --clusters=accelgor --time=5:00:00 image
scripts/hpc/train.sh --clusters=accelgor --time=1:00:00 occupancy
for scene in data/nerf/blender/*/; do
  scripts/hpc/train.sh --clusters=accelgor --time=4:00:00 nerf --data "$scene"
done
```

## FUTON ablations

Three studies on the occupancy task, each a config in `configs/ablation-futon/`
whose runs go to the folder of the same name in `logs/ablation-futon/`. All
keep the benchmark's settings (256^3 grid, 2000 epochs, lr 1e-2), so only the
model changes. The default model is the benchmark's FUTON: K=128 components,
CP rank 218 and a one-layer MLP decoder (131,673 parameters).

| Study | Models | What varies |
| --- | --- | --- |
| `basis` | `FUTON-<basis>`, 6 | cosine, lanczos, sinc, triangle, chebyshev and legendre, at the default size |
| `components_rank` | `FUTON-<basis>-K<K>-R<R>`, 48 | K and R over {32, 64, 128, 256}, for cosine, lanczos and sinc |
| `tensor_net` | `FUTON-<basis>` and `FUTON-<basis>-TR`, 6 | a tensor-ring combiner against the default CP one, at K=128, for cosine, lanczos and sinc |

A tensor ring of rank `r` has the size of a CP combiner of rank `r^2`; rank 15
gives 137,476 parameters, the closest to the CP model's 131,673. The CP models
of `tensor_net` repeat those of `basis`, so that each study stands alone. A run
takes about 25 s, so the 300 runs (60 models x 5 shapes) take about two hours:

```bash
scripts/hpc/train.sh --clusters=accelgor occupancy --config configs/ablation-futon/basis.yaml
scripts/hpc/train.sh --clusters=accelgor --time=3:00:00 occupancy \
  --config configs/ablation-futon/components_rank.yaml
scripts/hpc/train.sh --clusters=accelgor occupancy --config configs/ablation-futon/tensor_net.yaml
```

## Overrides

To try another value for one model, override it and send the runs elsewhere:

```bash
scripts/hpc/train.sh --clusters=accelgor --time=2:00:00 nerf \
  --models SIREN --data data/nerf/blender/lego \
  --set models.SIREN.train.lr=2.0e-4 --log-dir logs/siren-lr
```

After a model's entry in a config changes, `--overwrite` reruns it in place:

```bash
scripts/hpc/train.sh --clusters=accelgor --time=2:00:00 image \
  --models Gauss --overwrite
```

## Reports

Training writes each run's `results.json`, logs and `checkpoint.pt`, but no
figures. Tables, plots and examples are made afterwards from the pulled logs:

```bash
scripts/hpc/pull_logs.sh                # on the local machine
python scripts/report_image.py          # -> results/image/
python scripts/report_occupancy.py      # -> results/occupancy/
python scripts/report_nerf.py           # -> results/nerf/
python scripts/report_ablation.py       # -> results/ablation-futon/<study>/
```

Each task report writes:

- a table of the models (CSV, Markdown and LaTeX): their size, training time,
  inference speed and each metric averaged over the signals as `mean±std`,
  then a super column per signal (image, with 24 signals, keeps the average);
- convergence plots of each metric against iteration and against time, as PDF
  and PGF;
- qualitative examples rebuilt from the checkpoints for the signals that
  `--qualitative` names (`none` skips them), each signal's also composed into
  one `comparison.pdf`.

The ablation report writes the same table for each study, with convergence
plots for `basis`, IoU against R at each K and against K at each R for
`components_rank`, and CP against TR for each basis for `tensor_net`.

Rendering a mesh needs a display, so run the occupancy report on a
workstation. NeRF renders need none and can run on the cluster:

```bash
sbatch --clusters=accelgor --time=1:00:00 --gpus-per-node=1 --cpus-per-task=8 \
  --chdir=$VSC_DATA/projects/neurofield --output=logs/slurm/%x-%j.out \
  --job-name=nf-report --wrap \
  "source $VSC_DATA/venvs/neurofield-env/bin/activate && python scripts/report_nerf.py"
```

To read the runs directly instead:

```python
records = [json.loads(p.read_text()) for p in Path("logs").rglob("results.json")]
table = pd.json_normalize(records)  # task, data, model, num_params, metrics.*, setup.*
```
