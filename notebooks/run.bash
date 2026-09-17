#!/bin/bash
# Run all Jupyter notebooks in-place.
# Requires: pip install papermill  (one-time setup)

set -e

DIR="$(cd "$(dirname "$0")" && pwd)"

# Activate deepenv conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deepenv

# Ensure papermill is installed
python -c "import papermill" 2>/dev/null || pip install -q papermill

NOTEBOOKS=(
    "image_representation.ipynb"
    "occupancy_volume.ipynb"
    "nerf.ipynb"
    # "image_super_resolution.ipynb"
    # "image_denoising.ipynb"
    # "ct_reconstruction.ipynb"
    # "mri_reconstruction.ipynb"
)

for nb in "${NOTEBOOKS[@]}"; do
    # Kill all GPU compute processes and free memory
    echo "Clearing GPU memory..."
    nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null || true
    sleep 2
    echo "========================================"
    echo "Running: $nb"
    echo "========================================"
    papermill "$DIR/$nb" "$DIR/$nb" --cwd "$DIR"
    echo ""
done

echo "All notebooks finished."
