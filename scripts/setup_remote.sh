#!/usr/bin/env bash
# Bootstrap the neurofield repo on a fresh GPU machine (RunPod pod, VSC node, ...).
# Run from anywhere: bash scripts/setup_remote.sh
# Assumes: python + pip + a CUDA-enabled torch already present (e.g. RunPod PyTorch
# template or an HPC PyTorch module). Installs neurofield WITHOUT letting pip touch
# the platform's torch/numpy stack.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Container images (e.g. RunPod's) mark the system python "externally managed"
# (PEP 668) even though torch is installed in it; inside a disposable container
# installing alongside is fine. Harmless where the guard doesn't exist (VSC).
export PIP_BREAK_SYSTEM_PACKAGES=1

echo "==> Installing neurofield (editable, --no-deps to keep platform torch) + notebook extras"
pip install --no-deps -e . || exit 1
pip install kornia torchmetrics opt-einsum seaborn pandas matplotlib scipy pillow tqdm ipywidgets

echo "==> Fetching Kodak test images into data/Kodak (used via /scratch/data/Kodak/...)"
mkdir -p data/Kodak
for i in $(seq -w 1 24); do
  f="data/Kodak/kodim${i}.png"
  if [ ! -f "$f" ]; then
    curl -fsSL -o "$f" "https://r0k.us/graphics/kodak/kodak/kodim${i}.png" \
      || { rm -f "$f"; echo "   warning: kodim${i}.png download failed (continuing)"; }
  fi
done

echo "==> Mapping /scratch/data -> ${REPO_ROOT}/data (notebooks hardcode /scratch/data/...)"
if [ ! -e /scratch/data ]; then
  mkdir -p /scratch 2>/dev/null && ln -s "${REPO_ROOT}/data" /scratch/data 2>/dev/null \
    || echo "   warning: cannot create /scratch/data (no permission on this machine)." \
            " Edit IMAGE_PATH in the notebook to '../data/Kodak/kodim19.png' instead."
else
  echo "   /scratch/data already exists — leaving it alone."
  [ -e /scratch/data/Kodak ] || cp -r data/Kodak /scratch/data/ 2>/dev/null || true
fi

echo "==> Sanity check"
python - <<'PY'
import os, torch
print("torch:", torch.__version__, "| CUDA:", torch.cuda.is_available(),
      "| device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE — CPU ONLY, do not train!")
import neurofield
print("neurofield imported from:", os.path.dirname(neurofield.__file__))
print("kodim19 reachable at /scratch/data/Kodak/kodim19.png:",
      os.path.exists("/scratch/data/Kodak/kodim19.png"))
PY
echo "==> Done. Open notebooks/image_representation.ipynb and run all cells."
