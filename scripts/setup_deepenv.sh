#!/usr/bin/env bash
# Create/refresh the `deepenv` virtualenv on the RunPod pod and install
# neurofield into it in editable mode, registered as Jupyter kernel "deepenv"
# (the kernel name the repo's notebooks reference).
# The venv lives on /workspace so it survives pod stop/restart; the kernel
# registration lands on the container disk, so re-run this script after a
# pod restart (it is idempotent and takes seconds when the venv exists).
# Usage (on the pod): bash scripts/setup_deepenv.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV=/workspace/deepenv

if [ ! -x "$VENV/bin/python" ]; then
  echo "==> Creating venv at $VENV (inheriting system torch/CUDA)"
  if ! python3 -m venv "$VENV" --system-site-packages 2>/dev/null; then
    apt-get update -qq && apt-get install -y -qq python3-venv
    python3 -m venv "$VENV" --system-site-packages
  fi
fi

echo "==> Installing neurofield (editable) + notebook extras into deepenv"
"$VENV/bin/pip" install -q --no-deps -e "$REPO_ROOT"
"$VENV/bin/pip" install -q kornia torchmetrics opt-einsum seaborn pandas \
  matplotlib scipy pillow tqdm ipywidgets ipykernel

echo "==> Registering Jupyter kernel 'deepenv'"
"$VENV/bin/python" -m ipykernel install --name deepenv --display-name "Python (deepenv)" >/dev/null

# Cap CPU threads in the kernel env: on many-core cloud hosts, PyTorch's
# default (one OpenMP thread per core, e.g. 96) makes the small CPU-side
# ops in the training loop ~25x slower than a modest thread count.
"$VENV/bin/python" - <<'PY'
import json, glob
for p in glob.glob('/usr/local/share/jupyter/kernels/deepenv/kernel.json'):
    k = json.load(open(p))
    k.setdefault('env', {})['OMP_NUM_THREADS'] = '8'
    json.dump(k, open(p, 'w'), indent=1)
PY

echo "==> Sanity check"
"$VENV/bin/python" - <<'PY'
import sys, os, torch, neurofield
print("python :", sys.executable)
print("torch  :", torch.__version__, "| CUDA:", torch.cuda.is_available())
print("nf from:", os.path.dirname(neurofield.__file__))
PY
echo "==> Done. Jupyter: pick kernel 'deepenv'. VS Code: select interpreter /workspace/deepenv/bin/python."
