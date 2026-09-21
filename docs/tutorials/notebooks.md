# Notebooks

Three notebooks in `notebooks/` run one signal of each benchmark end to end: every model at equal size, the learning curves, and a results table. They are the interactive version of the training scripts and the fastest way to see how the models compare on your own data.

| Notebook | Signal | Models | Runtime on a GPU |
| --- | --- | --- | --- |
| `image_representation.ipynb` | `data/Kodak/kodim17.png` | 12, about 200k parameters each, 2000 epochs | about 8 minutes |
| `occupancy_volume.ipynb` | `data/occupancy/thai_statue.ply` at $256^3$ | 13, about 133k parameters each, 2000 epochs | about 4 minutes |
| `nerf.ipynb` | `data/nerf/blender/lego` under FINER's protocol | 13, about 76k parameters each, 37,500 steps | about two hours (set `NUM_STEPS = 5_000` for a quick pass) |

Each notebook has the same shape: load the signal, define the model configurations (the constructor arguments and learning rate of each model, with `# deviation` comments where a setting departs from the authors' code), train every model with `nf.train` or `nf.nerf.train`, collect the histories into a `pandas.DataFrame`, and plot PSNR or IoU against iterations and against time with seaborn.

```bash
pip install -e ".[3d,nerf]" jupyterlab
python scripts/download_kodak.py --images 17
jupyter lab notebooks/
```

The notebooks read data from `../data/`, relative to their folder, and are committed without outputs; run them top to bottom. The NeRF notebook also renders an orbit video of each model with `imageio`, which the `nerf` extra provides.
