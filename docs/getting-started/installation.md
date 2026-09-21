# Installation

## Requirements

- Python 3.11 or newer.
- PyTorch 2.10 or newer, with or without CUDA. Everything runs on the CPU; a GPU makes training practical.
- Linux with a CUDA build of PyTorch also gives you Triton, which FUTON's fused sparse kernels use. Without it (CPU, or other platforms) the same computation runs with ordinary PyTorch operations.

## Install

The package is installed from the repository, in editable mode so that the scripts, configs and notebooks find it:

```bash
git clone https://github.com/pashtari/neurofield.git
cd neurofield
pip install -e .
```

Check the install:

```python
import torch, neurofield as nf

print(nf.__version__, torch.cuda.is_available())
```

## Extras

Some tasks need packages that the core does not:

| Extra | Installs | Needed for |
| --- | --- | --- |
| `3d` | PyMCubes, pycollada, trimesh, Open3D | Meshes as occupancy volumes (`OccupancyCoordinateDataset` on a mesh file), exporting and rendering reconstructed meshes |
| `nerf` | nerfacc, imageio | CUDA occupancy-grid ray marching (a pure-PyTorch renderer is built in), writing novel-view videos |
| `dev` | black, pytest, pyinstrument | Formatting, the test suite, profiling |
| `docs` | mkdocs-material, mkdocstrings | Building this site |

```bash
pip install -e ".[3d,nerf,dev]"
```

!!! note "nerfacc"
    nerfacc compiles a CUDA extension the first time it is used and needs a CUDA toolkit for that. Without one, `nf.nerf.is_nerfacc_available()` is `False` and `nf.nerf.create_renderer()` falls back to the PyTorch renderer, which produces the same images more slowly.

## Benchmark data

The experiments use three public datasets. The scripts download them into `data/`, which is ignored by git:

```bash
python scripts/download_kodak.py      # 24 Kodak images (768 x 512), about 30 MB
python scripts/download_meshes.py     # 5 Stanford meshes, about 3 GB
python scripts/download_blender.py    # 8 NeRF synthetic (Blender) scenes, about 2.4 GB
```

Each script takes `--output-dir`, `--force`, and a subset selector (`--images 1 17 23`, `--meshes armadillo thai_statue`, `--scenes lego ship`). Interrupted downloads resume; existing files are kept. The resulting layout is what the training scripts and notebooks expect:

```
data/
├── Kodak/kodim01.png ... kodim24.png
├── occupancy/armadillo.ply  dragon.ply  happy_buddha.ply  lucy.ply  thai_statue.ply
└── nerf/blender/<scene>/transforms_{train,val,test}.json + images
```

Occupancy volumes are voxelized on first use and cached beside the mesh in `data/occupancy/.cache/<name>_<resolution>.pt`.

## Development

```bash
pip install -e ".[dev]"
black --check neurofield tests scripts   # formatting, line length 88
pytest -q                                 # tests that need CUDA or local data skip themselves
```
