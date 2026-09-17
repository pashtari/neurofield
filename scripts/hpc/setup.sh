#!/usr/bin/env bash
# Prepare HPC-UGent for the benchmarks: clone or update the repository at
# $VSC_DATA/projects/neurofield, build the neurofield-env venv, and download the
# LPIPS weights and benchmark data. Run once from the local machine (it pushes
# local commits and connects over SSH) or on a login node. Safe to rerun, e.g.
# after dependency changes.
#     scripts/hpc/setup.sh
set -euo pipefail

if [[ -z ${VSC_DATA:-} ]]; then
  # Local machine: push local commits, then run this script on a login node,
  # which checks out the same commit.
  script=$(<"$0")
  cd "$(dirname "$0")"
  if [[ -n $(git status --porcelain --untracked-files=no) ]]; then
    echo "Commit your changes first; the cluster only gets pushed commits." >&2
    exit 1
  fi
  [[ -z $(git status --porcelain) ]] || echo "Note: untracked files are not synced." >&2
  git push --quiet
  exec ssh "${HPC_HOST:-hpc-ugent}" \
    "NEUROFIELD_BRANCH=$(git symbolic-ref --short HEAD)" \
    "NEUROFIELD_COMMIT=$(git rev-parse HEAD)" \
    "NEUROFIELD_ROOT=$(printf %q "${NEUROFIELD_ROOT:-}")" \
    "NEUROFIELD_VENV=$(printf %q "${NEUROFIELD_VENV:-}")" \
    bash -c "$(printf %q "$script")"
fi

ROOT=${NEUROFIELD_ROOT:-$VSC_DATA/projects/neurofield}
VENV=${NEUROFIELD_VENV:-$VSC_DATA/venvs/neurofield-env}
REPO=https://github.com/pashtari/neurofield.git

# uv's standalone Python and the PyPI wheels are generic x86-64, so one venv
# serves every cluster. Caches stay off the small home quota, and login nodes
# cap each process at 2 GB of virtual memory, so tools run serially.
export UV_CACHE_DIR="$VSC_SCRATCH/cache/uv"
export UV_PYTHON_INSTALL_DIR="$VSC_DATA/python"
export UV_LINK_MODE=copy
export UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_INSTALLS=1 RAYON_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
module --force purge >/dev/null 2>&1 || true

echo "==> Repository: $ROOT"
[[ -d $ROOT/.git ]] || git clone "$REPO" "$ROOT"
cd "$ROOT"
git fetch --quiet
[[ -z ${NEUROFIELD_BRANCH:-} ]] || git checkout --quiet "$NEUROFIELD_BRANCH"
git merge --ff-only --quiet '@{u}'
if [[ -n ${NEUROFIELD_COMMIT:-} && $(git rev-parse HEAD) != "$NEUROFIELD_COMMIT" ]]; then
  echo "The cluster is at $(git rev-parse --short HEAD), not ${NEUROFIELD_COMMIT:0:7}." >&2
  exit 1
fi
git log --oneline -1

echo "==> Environment: $VENV"
uv venv --allow-existing --python 3.12 "$VENV"
uv pip install --python "$VENV/bin/python" -e ".[3d,dev]"

# Runtime settings, applied whenever the venv is activated (jobs and shells).
if ! grep -q "# neurofield settings" "$VENV/bin/activate"; then
  cat >> "$VENV/bin/activate" <<'SETTINGS'

# neurofield settings
export XDG_CACHE_HOME="$VSC_SCRATCH/cache"
export TORCH_HOME="$VSC_SCRATCH/cache/torch"
export TRITON_CACHE_DIR="$VSC_SCRATCH/cache/triton"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
SETTINGS
fi
source "$VENV/bin/activate"

echo "==> LPIPS weights: $TORCH_HOME"
weights="$TORCH_HOME/hub/checkpoints"
mkdir -p "$weights"
for file in vgg16-397923af.pth alexnet-owt-7be5be79.pth; do
  [[ -f $weights/$file ]] || curl -fsSL -o "$weights/$file" "https://download.pytorch.org/models/$file"
done

echo "==> Data: $ROOT/data (existing files are kept)"
python scripts/download_kodak.py
python scripts/download_meshes.py
python scripts/download_blender.py --workers 4

echo "==> Done. Test with:"
echo "    scripts/hpc/train.sh --clusters=accelgor --time=0:15:00 image --data data/Kodak/kodim01.png --models SIREN"
