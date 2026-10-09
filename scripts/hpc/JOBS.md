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
| super_resolution | 100 DIV2K images x 16 models = 1600 | ~1.5 h per image | `logs/super_resolution/<image>/<model>/` |

```bash
scripts/hpc/train.sh --clusters=accelgor --time=5:00:00 image
scripts/hpc/train.sh --clusters=accelgor --time=1:00:00 occupancy
for scene in data/nerf/blender/*/; do
  scripts/hpc/train.sh --clusters=accelgor --time=4:00:00 nerf --data "$scene"
done
# One image per array task: %4a becomes the task id, zero-padded to four digits.
# An image takes about 85 minutes on one GPU and the tasks backfill singly, so
# a two-hour wall time starts them far sooner than one long job.
scripts/hpc/train.sh --clusters=accelgor --time=2:00:00 --array=801-900 super_resolution \
  --data data/DIV2K/DIV2K_valid_HR/%4a.png
```

## FUTON ablations

Four studies on the image task, each a config in `configs/ablation-futon/`
whose runs go to the folder of the same name in `logs/ablation-futon/`. All
keep the benchmark's settings (24 Kodak images, 2000 epochs on 10% of the
pixels, lr 3e-2), so only the model changes. The default model is the
benchmark's FUTON: K = (H/2, W/2) components, CP rank 224 and a one-layer MLP
decoder (194,435 parameters).

| Study | Models | What varies |
| --- | --- | --- |
| `basis` | `FUTON-<basis>`, 6 | cosine, lanczos, sinc, triangle, chebyshev and legendre, at the default size |
| `components_rank` | `FUTON-<basis>-a<alpha>-b<beta>`, 50 | K = (alpha H, alpha W) and R = beta min(K), both over {1/8, 1/4, 1/2, 1, 2}, for lanczos and sinc |
| `tensor_net` | `FUTON-<basis>` and `FUTON-<basis>-TR`, 4 | a tensor-ring combiner against the default CP one, for lanczos and sinc |
| `decoder` | `FUTON-<basis>` and `-linear`, 4 | the paper's linear decoder against the benchmark's MLP, at equal size |

A tensor ring of rank `r` has the size of a CP combiner of rank `r^2`; rank 15
gives 195,528 parameters, the closest to the CP model's 194,435. The MLP
decoder holds 51,075 of those parameters, 26% of the model, which a linear
decoder returns to the combiner as CP rank 302 (194,189 parameters). The CP
models of `tensor_net` and `decoder` repeat those of `basis`, so that each
study stands alone. One image per job keeps the jobs short:

```bash
# Every run costs about 20 s on top of its training (data, model and LPIPS
# set-up, evaluation), so the basis study takes close to two hours over the 24
# images with its two slow polynomial bases, the other two about an hour.
for study in basis tensor_net decoder; do
  scripts/hpc/train.sh --clusters=accelgor --time=2:30:00 image --config configs/ablation-futon/$study.yaml
done
# The components grid holds models up to 9.4M parameters, so one image per task.
scripts/hpc/train.sh --clusters=accelgor --time=1:00:00 --array=1-24 image \
  --config configs/ablation-futon/components_rank.yaml --data data/Kodak/kodim%2a.png
scripts/hpc/train.sh --clusters=accelgor --time=1:00:00 nerf \
  --config configs/ablation-futon/components_nerf.yaml --data data/nerf/blender/lego
```

## Error bound

`scripts/futon_bound.py` runs the error-bound study of the docs' concepts
page on the 24 Kodak images and the 5 occupancy volumes: in the sinc basis,
at every spectral resolution and rank, it bounds the best error of FUTON
with a linear decoder, builds the FUTON that attains the upper bound, and
trains the same FUTON with Adam; then, on every signal at K = N/2 and one
rank, it trains for five times the epochs and runs least squares on the
whole grid. It records into `logs/bound/` and is resumable. One A100 does it
all in about two hours, bound and Adam in the same job, so that the cost of
the two is compared under the same conditions.

```bash
sbatch --clusters=accelgor --gpus-per-node=1 --cpus-per-task=8 \
  --time=3:00:00 --chdir=$VSC_DATA/projects/neurofield \
  --job-name=nf-bound --output=logs/slurm/%x-%j.out --wrap \
  "source $VSC_DATA/venvs/neurofield-env/bin/activate && python scripts/futon_bound.py"
```

Pull `logs/bound/` with `pull_logs.sh` and run `python scripts/report_paper.py
--tasks bound` locally for the figures and tables in `results/bound/`, which
take seconds and no GPU.

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
shapes; for NeRF the scenes below; for super-resolution the two example
images) and its accuracy from the sweeps.

```bash
common="--clusters=accelgor --exclusive --gpus-per-node=1 --chdir=$VSC_DATA/projects/neurofield --output=logs/slurm/%x-%j.out"
sbatch $common --time=5:00:00 --job-name=nf-timing-2d --wrap \
  "source $VSC_DATA/venvs/neurofield-env/bin/activate \
   && python scripts/profile_speed.py \
   && python scripts/train_image.py --overwrite --log-dir logs/timing/image \
   && python scripts/train_occupancy.py --overwrite --log-dir logs/timing/occupancy"
# An exclusive job waits for a whole node to drain, and short jobs backfill far
# sooner than long ones (a 4 h job sat two days without a planned start), so each
# NeRF scene runs as two balanced model groups of about an hour of training each.
# All four jobs write disjoint runs into the same logs/timing/nerf.
A="Instant-NGP Gauss GA-Planes FINER TensoRF-CP FUTON-lanczos"
B="MFN WIRE TensoRF-VM SIREN PE-MLP FUTON-sinc RFF"
for scene in lego hotdog; do
  sbatch $common --time=1:30:00 --job-name=nf-timing-nerf-$scene-a --wrap \
    "source $VSC_DATA/venvs/neurofield-env/bin/activate \
     && python scripts/train_nerf.py --data data/nerf/blender/$scene --models $A --overwrite --log-dir logs/timing/nerf"
  sbatch $common --time=1:30:00 --job-name=nf-timing-nerf-$scene-b --wrap \
    "source $VSC_DATA/venvs/neurofield-env/bin/activate \
     && python scripts/train_nerf.py --data data/nerf/blender/$scene --models $B --overwrite --log-dir logs/timing/nerf"
done
# Super-resolution: the two example images, one exclusive job each.
for image in 0882 0801; do
  sbatch $common --time=2:00:00 --job-name=nf-timing-sr-$image --wrap \
    "source $VSC_DATA/venvs/neurofield-env/bin/activate \
     && python scripts/train_super_resolution.py --data data/DIV2K/DIV2K_valid_HR/$image.png --overwrite --log-dir logs/timing/super_resolution"
done
# Its inference profile reads the example image's runs, so it follows the sweep.
sbatch $common --time=0:30:00 --job-name=nf-speed-sr --wrap \
  "source $VSC_DATA/venvs/neurofield-env/bin/activate && python scripts/profile_speed.py --tasks super_resolution"
```

Peak memory is a property of the process, not of the node, so one shared job
measures it on each task's profiled signal: every training run records the
most a step allocated beyond the data and parameters already on the GPU, and
a short run reaches that peak (the radiance fields early, while the occupancy
grid is still dense), and `profile_speed.py` records the peak of each
inference. Both land in `logs/memory/`, apart from the clean timings, and the
report takes the memory columns and figures from there.

```bash
shared="--clusters=accelgor --gpus-per-node=1 --chdir=$VSC_DATA/projects/neurofield --output=logs/slurm/%x-%j.out"
sbatch $shared --time=2:00:00 --job-name=nf-memory --wrap \
  "source $VSC_DATA/venvs/neurofield-env/bin/activate \
   && python scripts/train_image.py --data data/Kodak/kodim01.png --set train.num_epochs=100 --overwrite --log-dir logs/memory/image \
   && python scripts/train_occupancy.py --data data/occupancy/thai_statue.ply --set train.num_epochs=100 --overwrite --log-dir logs/memory/occupancy \
   && python scripts/train_nerf.py --data data/nerf/blender/lego --set train.num_steps=2500 --overwrite --log-dir logs/memory/nerf \
   && python scripts/train_super_resolution.py --data data/DIV2K/DIV2K_valid_HR/0882.png --set train.num_epochs=100 --overwrite --log-dir logs/memory/super_resolution \
   && python scripts/profile_speed.py --out-dir logs/memory"
```

`report_paper.py` writes everything the paper needs into `results/`, and
nothing else. Per task, in `results/<task>/` so that each is a LaTeX
subfigure:

- quality against training time, for FUTON and the strongest model of each
  other family, the band one standard error, as in the tables; time is
  logarithmic, where the curves span more than a decade of it, and the metric
  linear;
- every model's final quality against its training time, its inference
  throughput, and its peak training and inference memory, all axes linear;
- one table of every model's size, the cost of both phases and its mean final
  metrics, the best in bold and the second underlined. The averages carry one
  standard error over the signals, taken once each signal's own level is
  removed, but for IoU, which saturates on every shape. The occupancy table also gives each shape's IoU, and the NeRF one
  every scene's metrics under a super column, which takes a page turned
  sideways;
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

The FUTON ablations go to `results/ablation/`: `basis`, `combiner` and
`decoder` each as a table and a convergence plot, the tensor ring against CP
and the linear decoder against the MLP for both bases, and the components
study as PSNR against the rank fraction at each component fraction and the
reverse, one figure per basis, for the fractions 1/4 to 2.

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
