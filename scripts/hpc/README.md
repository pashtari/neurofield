# Running the benchmarks on HPC-UGent

These scripts run `scripts/train_<task>.py` as Slurm jobs on the HPC-UGent GPU
clusters. They connect through the SSH host `hpc-ugent` (override with
`HPC_HOST`) and use the repository at `$VSC_DATA/projects/neurofield` and the
venv at `$VSC_DATA/venvs/neurofield-env`.

| File | Run from | Purpose |
| --- | --- | --- |
| `setup.sh` | local machine, once | Prepares the cluster: repository, venv, data, LPIPS weights |
| `train.sh` | local machine or login node | Submits one task as a GPU job |
| `pull_logs.sh` | local machine | Copies `logs/` from the cluster |
| `JOBS.md` | — | Every job of this work: benchmarks and ablations |

The cluster runs committed code only. Run from the local machine, `setup.sh`
and `train.sh` first push local commits to GitHub, then bring the cluster's
repository to the same branch and commit; they stop if that fails. They refuse
to run while tracked files have uncommitted changes, so commit first.
Untracked files are not synced.

## 1. Connect to HPC-UGent

The cluster is reached over SSH with a VSC account and the key pair uploaded on
the [VSC account page](https://account.vscentrum.be). Add the login node to
`~/.ssh/config` on the local machine, as the host `hpc-ugent` these scripts
expect:

```
Host hpc-ugent
    HostName login.hpc.ugent.be
    User vscXXXXX                 # your VSC account
    IdentityFile ~/.ssh/id_rsa_vsc
    IdentitiesOnly yes
    ServerAliveInterval 60
    ServerAliveCountMax 5
```

Check it with `ssh hpc-ugent`, which also opens a shell on a login node. The
scripts accept another host name through `HPC_HOST`. Since each of their calls
connects anew, reusing one connection is worthwhile:

```
    ControlMaster auto
    ControlPath ~/.ssh/control-%r@%h:%p
    ControlPersist 10m
```

## 2. Set up the cluster (once)

From the local repository:

```bash
scripts/hpc/setup.sh
```

The script connects to `hpc-ugent` and runs on a login node. It takes about 10
minutes and does the following:

1. **Repository.** Clones the repository to `$VSC_DATA/projects/neurofield`
   if needed, and checks out the local branch and commit.
2. **Environment.** Creates the Python 3.12 venv `neurofield-env` in
   `$VSC_DATA/venvs/` with `uv`, and installs neurofield in editable mode with
   the `3d` and `dev` extras. The PyPI PyTorch wheels include their own CUDA
   libraries, so the same venv works on every GPU cluster, and no modules are
   needed. The venv's `activate` script also sets the cache directories
   (`$VSC_SCRATCH/cache`) and `OMP_NUM_THREADS`.
3. **LPIPS weights.** Caches the VGG and AlexNet backbones in
   `$VSC_SCRATCH/cache/torch`, because compute nodes cannot download them.
4. **Data.** Downloads the benchmark data into `data/`:
   - the 24 Kodak images to `data/Kodak/` (about 30 MB);
   - the 5 Stanford meshes to `data/occupancy/` (about 3 GB);
   - the 8 Blender scenes to `data/nerf/blender/` (about 2.4 GB).

The script is safe to rerun: existing files are kept. Rerun it after changing
dependencies in `pyproject.toml`. The venv takes about 7 GB of the 25 GB
`$VSC_DATA` quota. The data (about 4 GB) and the logs live in
`$VSC_SCRATCH/neurofield/` instead, linked from the repository as `data/` and
`logs/`; to set up a new clone that way:

```bash
mkdir -p $VSC_SCRATCH/neurofield/{data,logs}
ln -s $VSC_SCRATCH/neurofield/data $VSC_SCRATCH/neurofield/logs $VSC_DATA/projects/neurofield/
```

For an interactive shell on the cluster, run
`source $VSC_DATA/venvs/neurofield-env/bin/activate`. Login nodes limit each
process to 2 GB of virtual memory, which is not enough to import PyTorch, so
test the environment in a job (step 3).

## 3. Run the experiments

Submit a task from the local repository or from the repository on a login node:

```bash
scripts/hpc/train.sh [sbatch options] <task> [train_<task>.py options]
```

Run locally, `train.sh` pushes local commits, connects to `hpc-ugent`, and
reruns itself there. On the login node, it updates the repository (to the local
commit when called from the local machine, otherwise with a fast-forward pull)
and submits itself with `sbatch`. The job then trains the models and records
the commit in its output, e.g. `commit: 1a2b3c4`. Jobs are named
`nf-<task>[-<signal>][-<model>]`, naming whatever the job runs only one of, so
`squeue` and the log files say what is running. Pass `--job-name=` to override.
It also prints the scheduler's estimated start time, e.g.

```
Submitting nf-image-kodim01-FUTON-sinc at commit 1a2b3c4
Submitted batch job 15754794 on cluster accelgor
Estimated start: 2026-09-17T21:33:00
```

That estimate assumes every running job uses its full wall time, so jobs often
start earlier, and later when higher-priority jobs arrive.

The venv runs on two GPU clusters, accelgor (A100) and litleo (H100); the
third, joltik, has V100s, for which its CUDA 13 PyTorch wheels have no
kernels. Keep one cluster for a whole benchmark, since `results.json` records
training times. A list such as
`--clusters=accelgor,litleo` submits where the job can start soonest, which is
useful for quick tests. GPUs are often fully allocated, and then a short wall
time does not help: the job waits for one to be released. To see the wait
before submitting:

```bash
sbatch --clusters=accelgor --test-only --time=0:10:00 --gpus-per-node=1 \
  scripts/hpc/train.sh image
```

- **sbatch options** come before the task and use the `--name=value` form.
  The job requests 1 GPU, 8 CPUs, and 1 hour by default. Short jobs start
  sooner, since the scheduler can backfill them, so the commands below split
  each sweep into jobs that fit the hour. Raise it with `--time=` (accelgor
  allows up to 72 hours) to put more runs in one job.
- **Task options** (`--data`, `--models`, `--config`, `--set`, `--log-dir`,
  `--overwrite`) come after the task and are passed to
  `scripts/train_<task>.py`. Paths are relative to the repository root on the
  cluster. The runs of `configs/<name>.yaml` go to `logs/<name>/`. Several
  `--config` files are merged in order, the last naming the log folder, so a
  small config can hold only what changes, e.g.
  `--config configs/image.yaml changes.yaml`. Runs are keyed by model name,
  so give variants their own names or another `--log-dir`.

Measured on A100s (2000 epochs; NeRF 37,500 steps), a run takes about:

| Task | Per model | Per signal | Whole sweep |
| --- | --- | --- | --- |
| image | 37 s | 7.5 min (12 models) | 3 h (24 images) |
| occupancy | 19 s | 4 min (13 models) | 20 min (5 shapes) |
| nerf | 10 min | 2.2 h (13 models) | 18 h (8 scenes) |

Training a model takes about 17 s in the image and occupancy tasks; the image
figure is higher because of the 20 LPIPS evaluations. The NeRF figure includes
scoring the 200 test views, and was measured with eight jobs sharing nodes, so
a single job is faster.

Pending jobs run whichever code is in the repository when they start, so a
later `train.sh` call with new commits also affects jobs that are still queued.
The `commit:` line at the top of a job's output names the commit it started
from.

A quick test (a few minutes):

```bash
scripts/hpc/train.sh --clusters=accelgor --time=0:15:00 image \
  --data data/Kodak/kodim01.png --models SIREN
```

`JOBS.md` lists every job of this work: the three benchmark sweeps and the
FUTON ablations. Smaller jobs are scheduled sooner, so the sections below split
the benchmark sweeps further. Runs already finished are skipped, so the forms
mix freely.

### Image representation

All 12 models on the 24 Kodak images, written to `logs/image/<image>/<model>/`.
One job per image (about 9 minutes each) instead of one 3.5-hour job:

```bash
for image in data/Kodak/*.png; do
  scripts/hpc/train.sh --clusters=accelgor image --data "$image"
done
```

### Occupancy

All 13 models on the 5 Stanford shapes, written to
`logs/occupancy/<shape>/<model>/`. One job per shape (about 5 minutes each):

```bash
for shape in data/occupancy/*.ply; do
  scripts/hpc/train.sh --clusters=accelgor occupancy --data "$shape"
done
```

### NeRF

All 13 models on the 8 Blender scenes, with 37,500 steps per model. This is by
far the longest task: one job per scene and model (about 10 minutes each, 104
jobs) queues better than one job per scene. Runs are written to
`logs/nerf/<scene>/<model>/`:

```bash
models="RFF PE-MLP MFN SIREN Gauss WIRE FINER Instant-NGP TensoRF-CP TensoRF-VM
        GA-Planes FUTON-sinc FUTON-lanczos"
for scene in data/nerf/blender/*/; do
  for model in $models; do
    scripts/hpc/train.sh --clusters=accelgor nerf --data "$scene" --models "$model"
  done
done
```

Submitting 104 jobs opens 104 SSH connections; either set up `ControlMaster`
(step 1) or run the loop on a login node.

### Splitting and resuming

Runs that already have `results.json` are skipped, so a job that stops early
(a time limit kills it mid-run, or a node fails) is simply submitted again. To
spread a task over several GPUs, submit jobs with disjoint `--data` or
`--models` lists; jobs that share a signal and model would write to the same
run directory.

```bash
scripts/hpc/train.sh --clusters=accelgor --time=2:00:00 image --data data/Kodak/kodim0{1..9}.png
scripts/hpc/train.sh --clusters=accelgor --time=3:00:00 image --data data/Kodak/kodim{10..24}.png
```

## 4. Monitor and collect results

On the cluster:

```bash
squeue --clusters=accelgor -u $USER -o "%.10i %.30j %.2t %.9M %.9l"  # jobs
tail -f logs/slurm/nf-image-kodim01-FUTON-sinc-<jobid>.out           # output
scancel --clusters=accelgor <jobid>                                  # cancel
squeue --clusters=accelgor -u $USER --start          # estimated start times
```

The default `squeue` format truncates the name to 8 characters, hence the
explicit `-o` above.

A failed run leaves `error.txt` in its run directory. The job's final line
lists all failed runs.

To copy the logs to the local repository (`-n` is a dry run):

```bash
scripts/hpc/pull_logs.sh
```

It never deletes local files, and keeps a local file that is newer than the
cluster's copy, so delete a local run directory to take the cluster's again.

Each run directory contains `results.json` (settings, parameter count,
training time, final metrics, and training history), `log.txt`, `log.json`,
`checkpoint.pt`, and the reconstruction of an image or NeRF's selected test
views. Occupancy runs write no mesh; the report rebuilds one from
`checkpoint.pt` and renders it offscreen. `JOBS.md` shows how
`report_paper.py` turns the runs into the paper's figures and tables; to load
the runs directly:

```python
records = [json.loads(p.read_text()) for p in Path("logs").rglob("results.json")]
table = pd.json_normalize(records)
```
