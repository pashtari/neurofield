# Jobs

Every job of this work, all on one A100 of accelgor. `README.md` covers the
setup and how `train.sh` works. Each command below submits one job (the NeRF
loop, one per scene). Finished runs are skipped, so a job that stops early is
simply submitted again; `--overwrite` reruns finished runs instead.

## Benchmarks

Every model of `configs/<task>.yaml` on every signal of the task:

| Task | Runs | Time | Logs |
| --- | --- | --- | --- |
| image | 24 Kodak images x 12 models = 288 | ~3.5 h | `logs/image/<image>/<model>/` |
| occupancy | 5 Stanford shapes x 13 models = 65 | ~25 min | `logs/occupancy/<shape>/<model>/` |
| nerf | 8 Blender scenes x 13 models = 104 | ~2.5 h per scene | `logs/nerf/<scene>/<model>/` |

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
| `components_rank` | `FUTON-<basis>-K<K>-R<R>`, 32 | K and R over {32, 64, 128, 256}, for lanczos and sinc |
| `tensor_net` | `FUTON-<basis>` and `FUTON-<basis>-TR`, 4 | a tensor-ring combiner against the default CP one, at K=128, for lanczos and sinc |
| `decoder` | `FUTON-<basis>`, `-linear` and `-linear-R218`, 6 | the paper's linear decoder against the benchmark's MLP, at equal size and at equal rank |

A tensor ring of rank `r` has the size of a CP combiner of rank `r^2`; rank 15
gives 137,476 parameters, the closest to the CP model's 131,673. The MLP
decoder holds 47,961 of those parameters, 36% of the model, which a linear
decoder returns to the combiner as CP rank 342 (131,671 parameters); rank 218
is kept as well, to separate the parameters from the nonlinearity. The CP
models of `tensor_net` and `decoder` repeat those of `basis`, so that each
study stands alone. A run takes about 25 s, so the 240 runs (48 models x 5
shapes) take about 1.7 hours:

```bash
scripts/hpc/train.sh --clusters=accelgor occupancy --config configs/ablation-futon/basis.yaml
scripts/hpc/train.sh --clusters=accelgor --time=2:00:00 occupancy \
  --config configs/ablation-futon/components_rank.yaml
scripts/hpc/train.sh --clusters=accelgor occupancy --config configs/ablation-futon/tensor_net.yaml
scripts/hpc/train.sh --clusters=accelgor occupancy --config configs/ablation-futon/decoder.yaml
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
python scripts/report_paper.py          # -> results/, the paper's figures
python scripts/report_paper.py --no-panels   # without the renders, which need a GPU
```

Timings from the sweeps carry whatever else shared the node: on a clean node
Instant-NGP trained 1.7x faster, MFN and TensoRF-VM 1.2x, the rest within a
few percent. So the sweeps give the accuracy and separate exclusive jobs give
every time: `profile_speed.py` times inference on one signal per task, and
the timing runs retrain every model alone, after a warm-up pass that keeps
kernel compilation out of the clock, into `logs/timing/<task>/`. The report
takes its times from there wherever such a run exists (all 24 images and 5
shapes; for NeRF the scenes below) and its accuracy from the sweeps.

```bash
common="--clusters=accelgor --exclusive --gpus-per-node=1 --chdir=$VSC_DATA/projects/neurofield --output=logs/slurm/%x-%j.out"
sbatch $common --time=5:00:00 --job-name=nf-timing-2d --wrap \
  "source $VSC_DATA/venvs/neurofield-env/bin/activate \
   && python scripts/profile_speed.py \
   && python scripts/train_image.py --overwrite --log-dir logs/timing/image \
   && python scripts/train_occupancy.py --overwrite --log-dir logs/timing/occupancy"
sbatch $common --time=4:00:00 --job-name=nf-timing-nerf --wrap \
  "source $VSC_DATA/venvs/neurofield-env/bin/activate \
   && python scripts/train_nerf.py --data data/nerf/blender/lego data/nerf/blender/hotdog --overwrite --log-dir logs/timing/nerf"
```

`report_paper.py` writes everything the paper needs into `results/`, and
nothing else. Per task, in `results/<task>/` so that each is a LaTeX
subfigure:

- quality against training time, for FUTON and the strongest model of each
  other family, the band one within-signal standard error; time is logarithmic,
  where the curves span more than a decade of it, and the metric linear;
- every model's final quality against its training time, and against its
  inference rate, both axes linear;
- one table of every model's size, training time and final metrics, the best in
  bold and the second underlined. The averages carry one standard error over
  the signals, taken once each signal's own level is removed. The occupancy
  table also gives each shape's IoU, and the NeRF one every scene's metrics
  under a super column, which takes a page turned sideways;
- for each of the task's example signals, that signal with two regions boxed,
  each in its own colour, and those regions magnified for the ground truth and
  every featured model, framed in the colour of their box. The regions are
  found, not set: the two squares of fine detail where FUTON gains most
  squared error over the strongest baseline, which puts the magnifications
  where the models actually differ. `ZONES` restricts the search to a
  rectangle per box where a signal's telling regions are not the ones the
  search would reach, and a NeRF scene is rendered from the view where FUTON
  gains most of those that show it whole, which every run records.

The magnified panels are rebuilt from the checkpoints, which needs the signals
in `data/` and a GPU, so they are kept in `<task>/panels/<signal>/` and reused.
`--overwrite` draws them again, which a changed NeRF view calls for, and
`--no-panels` leaves them out. Meshes render offscreen, so no display is
needed; a shape file that differs from the cluster's would rebuild the wrong
mesh, which the report catches by recomputing each run's IoU.

The figures carry one TensoRF, the default variant for the dimension: CP in 2D,
where it is the only one, and VM in 3D. Four hues are as many as stay apart
under every kind of colour blindness, so a hue names a family and the line
style names the member, the stronger of a pair solid: SIREN and FINER share
the periodic-activation hue, as do the two FUTON bases. A shared `legend.pdf`
sits beside the task folders.

The FUTON ablations go to `results/ablation/`: `basis` as a table and a
convergence plot, `combiner_decoder` as one table of the tensor ring and the
linear decoder against the benchmark's CP and MLP, `tensor_net` as a
convergence plot, and the components study as IoU against R at each K and
against K at each R, one figure per basis.

`--orbit 60` also saves an orbit GIF of each NeRF model beside its panel, for
a talk. It costs about a minute a scene and the paper takes none of them, so
it is off by default.

The whole report runs on the cluster too, where the data sits beside the runs:

```bash
sbatch --clusters=accelgor --time=1:00:00 --gpus-per-node=1 --cpus-per-task=8 \
  --chdir=$VSC_DATA/projects/neurofield --output=logs/slurm/%x-%j.out \
  --job-name=nf-report --wrap \
  "source $VSC_DATA/venvs/neurofield-env/bin/activate && python scripts/report_paper.py"
```

To read the runs directly instead:

```python
records = [json.loads(p.read_text()) for p in Path("logs").rglob("results.json")]
table = pd.json_normalize(records)  # task, data, model, num_params, metrics.*, setup.*
```
