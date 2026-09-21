# Experiments

The paper's experiments are one protocol applied to three tasks, and this section documents them in enough detail to rerun them.

- [Benchmarks](benchmarks.md): the protocol, and every model's size, training time and final metrics on Kodak images, Stanford occupancy volumes and Blender radiance fields, with convergence plots and qualitative comparisons.
- [Ablations](ablations.md): FUTON's basis, the number of components against the rank, the combiner, and the decoder.
- [Reproducing the paper](reproduce.md): the data, the training scripts and their configs, the run directories, the report script that turns runs into figures and tables, and the Slurm scripts for a cluster.

## The protocol in one paragraph

Every model of a task is given the same parameter budget (about 195k for images, 132k for volumes and 75k for the density network of a radiance field), the hyperparameters of its authors' code wherever the budget allows them, and a learning rate from the grid {3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4} chosen on the task. Images and volumes train for 2000 Adam steps on random subsets of the signal, with cosine annealing to a hundredth of the learning rate; radiance fields for 37,500 steps of 4096 rays with exponential decay to a tenth, torch-ngp's recipe. Every run uses seed 0, and all runs of a task ran on the same A100 cluster. Metrics are computed on the full signal (all pixels, all voxels, all 200 test views), and the tables report the mean over the signals of a task with a within-signal standard error, the best value in bold and the second in italics.
