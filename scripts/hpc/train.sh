#!/bin/bash
#SBATCH --job-name=neurofield
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#
# Train the models of one benchmark task on one GPU of HPC-UGent.
#     scripts/hpc/train.sh [sbatch options] <image|occupancy|nerf> [train_<task>.py options]
#     scripts/hpc/train.sh --clusters=accelgor image --models SIREN FINER
# sbatch options come before the task and use the --name=value form. The
# default hour covers one image, one shape, or one NeRF model; short jobs are
# scheduled sooner, and finished runs are skipped when a job is resubmitted.
#
# The script runs in three places:
#   local machine: pushes local commits, connects over SSH, and reruns itself
#                  on a login node;
#   login node:    checks out the latest commit and submits itself with sbatch;
#   Slurm job:     trains the models.
set -euo pipefail

if [[ -z ${VSC_DATA:-} ]]; then
  cd "$(dirname "$0")"
  if [[ -n $(git status --porcelain --untracked-files=no) ]]; then
    echo "Commit your changes first; the cluster only gets pushed commits." >&2
    exit 1
  fi
  [[ -z $(git status --porcelain) ]] || echo "Note: untracked files are not synced." >&2
  git push --quiet
  root=${NEUROFIELD_ROOT:-'$VSC_DATA/projects/neurofield'}
  exec ssh "${HPC_HOST:-hpc-ugent}" \
    "NEUROFIELD_BRANCH=$(git symbolic-ref --short HEAD)" \
    "NEUROFIELD_COMMIT=$(git rev-parse HEAD)" \
    "NEUROFIELD_VENV=$(printf %q "${NEUROFIELD_VENV:-}")" \
    "$root/scripts/hpc/train.sh ${*@Q}"
fi

if [[ -z ${SLURM_JOB_ID:-} ]]; then
  root=$(cd "$(dirname "$0")/../.." && pwd)
  if [[ -z ${NEUROFIELD_SYNCED:-} ]]; then
    # Update the repository (to the local commit when called from the local
    # machine), then rerun the updated script.
    cd "$root"
    git fetch --quiet
    [[ -z ${NEUROFIELD_BRANCH:-} ]] || git checkout --quiet "$NEUROFIELD_BRANCH"
    git merge --ff-only --quiet '@{u}'
    if [[ -n ${NEUROFIELD_COMMIT:-} && $(git rev-parse HEAD) != "$NEUROFIELD_COMMIT" ]]; then
      echo "The cluster is at $(git rev-parse --short HEAD), not ${NEUROFIELD_COMMIT:0:7}." >&2
      exit 1
    fi
    NEUROFIELD_SYNCED=1 exec "$root/scripts/hpc/train.sh" "$@"
  fi

  sbatch_args=()
  while [[ $# -gt 0 && $1 == -* ]]; do
    sbatch_args+=("$1")
    shift
  done
  if [[ $# -eq 0 ]]; then
    echo "usage: $0 [sbatch options] <image|occupancy|nerf> [options]" >&2
    exit 2
  fi
  # Name the job nf-<task>[-<signal>][-<model>], naming only what is unique,
  # so that squeue and logs/slurm/<name>-<jobid>.out say what is running.
  name=nf-$1
  for option in data models; do
    values=()
    collect=""
    for arg in "$@"; do
      case $arg in
        --$option) collect=1 ;;
        --$option=*) values+=("${arg#*=}") ;;
        -*) collect="" ;;
        *) [[ -n $collect ]] && values+=("$arg") ;;
      esac
    done
    if [[ ${#values[@]} -eq 1 ]]; then
      value=${values[0]%/}  # NeRF scenes are directories
      value=${value##*/}
      name+=-${value%.*}
    fi
  done

  mkdir -p "$root/logs/slurm"
  echo "Submitting $name at commit $(git -C "$root" rev-parse --short HEAD)"
  submitted=$(sbatch --chdir="$root" --job-name="$name" "${sbatch_args[@]}" \
    "$root/scripts/hpc/train.sh" "$@")
  echo "$submitted"

  # "Submitted batch job <id> on cluster <name>". The scheduler needs a moment
  # to estimate the start of a pending job, so ask twice. The estimate assumes
  # every running job uses its full wall time, so jobs often start earlier --
  # and later when higher-priority ones arrive.
  read -r job cluster <<<"$(awk '{print $4, $7}' <<<"$submitted")"
  queue=(squeue ${cluster:+--clusters=$cluster} -j "$job" -h)
  state="" reason="" start=""
  for _ in 1 2; do
    read -r state reason start <<<"$("${queue[@]}" -o "%T %r %S" 2>/dev/null || true)"
    [[ $state == RUNNING || ($state == PENDING && -n $start && $start != N/A) ]] && break
    sleep 5
  done
  case $state in
    RUNNING) echo "Running on $("${queue[@]}" -o "%R")" ;;
    PENDING) if [[ -n $start && $start != N/A ]]; then
        echo "Estimated start: $start (waiting for $reason)"
      else
        echo "Pending ($reason); for an estimate: squeue -j $job --start"
      fi ;;
  esac
  exit 0
fi

task=${1:?usage: sbatch scripts/hpc/train.sh <image|occupancy|nerf> [options]}
shift

# Modules loaded on the login node would shadow the venv's libraries.
module --force purge >/dev/null 2>&1 || true
source "${NEUROFIELD_VENV:-$VSC_DATA/venvs/neurofield-env}/bin/activate"

echo "commit: $(git describe --always --dirty), node: $(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python "scripts/train_$task.py" "$@"
