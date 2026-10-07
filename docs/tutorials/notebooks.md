# Notebooks

Five notebooks in `notebooks/` run one signal end to end. Four compare every model of a task at equal size, with the learning curves and a results table, and the fifth compares the ways to fit FUTON. They are the interactive version of the training scripts and the fastest way to see how the models compare on your own data.

| Notebook | Signal | Models | Runtime on an RTX 4060 Ti |
| --- | --- | --- | --- |
| `image_representation.ipynb` | `data/Kodak/kodim17.png` | 12, about 200k parameters each, 2000 epochs | about 9 minutes |
| `occupancy_volume.ipynb` | `data/occupancy/thai_statue.ply` at $256^3$ | 13, about 133k parameters each, 2000 epochs | about 8 minutes |
| `nerf.ipynb` | `data/nerf/blender/lego` under FINER's protocol | 13, about 76k parameters each, 37,500 steps | about five hours (set `NUM_STEPS = 5_000` for a quick pass) |
| `image_super_resolution.ipynb` | `data/DIV2K/DIV2K_valid_HR/0882.png` from its 4× bicubic downsampling | 12 fields and Deep Image Prior, about 500k parameters each, 2000 steps | about 85 minutes |
| `futon_solvers.ipynb` | `data/Kodak/kodim19.png` | 4 FUTONs, about 194k parameters each: an MLP or a linear decoder, fitted by Adam, alternating least squares or multiplicative updates, 2000 steps | about 2 minutes |

Each notebook has the same shape: load the signal, define the model configurations (the constructor arguments and learning rate of each model, with `# deviation` comments where a setting departs from the authors' code), train every model with `nf.train` or `nf.nerf.train`, collect the histories into a `pandas.DataFrame`, and plot PSNR or IoU against iterations and against time with seaborn. The super-resolution notebook also builds its forward model: the models never see the high-resolution image, and each step renders the pixels under random blocks of low-resolution pixels, downsamples them with the benchmark's bicubic kernel, and matches the low-resolution pixels.

```bash
pip install -e ".[3d,nerf]" jupyterlab
python scripts/download_kodak.py --images 17 19
python scripts/download_div2k.py
jupyter lab notebooks/
```

The notebooks read data from `../data/`, relative to their folder, and are committed with the outputs of the runs timed above, so they read without being run; rerun them top to bottom. The NeRF notebook also renders an orbit video of each model with `imageio`, which the `nerf` extra provides.
