# Jobs

Every run in this work, as submitted on HPC-UGent. See `README.md` for the
setup and for how `train.sh` works. All jobs use one A100 (accelgor), one job
at a time per line below; finished runs are skipped, so a command that stops
early can simply be repeated.

## Benchmarks

The 14 (image) or 15 (3D) models of `configs/<task>.yaml` on every signal.

| Job | Runs | Time | Logs |
| --- | --- | --- | --- |
| image | 24 images x 14 models = 336 | ~3.5 h | `logs/image/<image>/<model>/` |
| occupancy | 5 shapes x 15 models = 75 | ~25 min | `logs/occupancy/<shape>/<model>/` |
| nerf | 8 scenes x 15 models = 120 | ~2.5 h per scene | `logs/nerf/<scene>/<model>/` |

```bash
scripts/hpc/train.sh --clusters=accelgor --time=5:00:00 image
scripts/hpc/train.sh --clusters=accelgor --time=1:00:00 occupancy
for scene in data/nerf/blender/*/; do
  scripts/hpc/train.sh --clusters=accelgor --time=4:00:00 nerf --data "$scene"
done
```

## FUTON ablations

`configs/ablation_futon.yaml` holds 40 models on the occupancy task, which is
the cheapest of the three and the most sensitive to the basis. They all share
the benchmark's settings (256^3 grid, 2000 epochs, lr 1e-2), so only the model
changes. 40 models x 5 shapes = 200 runs, about 1.5 hours in total:

```bash
scripts/hpc/train.sh --clusters=accelgor --time=3:00:00 occupancy \
  --config configs/ablation_futon.yaml --log-dir logs/ablation-futon
```

The runs land in `logs/ablation-futon/<shape>/<model>/`, separate from the
benchmark, and cover three questions:

| Study | Models | What varies |
| --- | --- | --- |
| Reference | `FUTON-cosine`, `FUTON-sinc` | the benchmark entries, repeated (K=256, R=144, 131,617 parameters) |
| Rank and components | `FUTON-<basis>-K<K>-R<R>`, 32 models | K and R over {64, 128, 256, 512} for both bases (16,513 to 1,049,601 parameters) |
| Tensor ring | `FUTON-<basis>-TR-K256-R<r>`, 4 models | a TR combiner in place of CP, r = 12 and 16 |
| Basis | `FUTON-chebyshev`, `FUTON-legendre` | two more bases at the reference size |

A TR combiner of rank `r` has exactly the parameter count of a CP combiner of
rank `r^2`, so `TR-K256-R12` pairs with the reference `R=144` and
`TR-K256-R16` with `K256-R256`, each at equal size.

To run one study alone, add `--models`, e.g.
`--models FUTON-chebyshev FUTON-legendre`. To split the sweep over several
GPUs, give each job its own shapes with `--data`.

## Collecting the results

```bash
scripts/hpc/pull_logs.sh   # from the local machine
```

```python
records = [json.loads(p.read_text()) for p in Path("logs").rglob("results.json")]
table = pd.json_normalize(records)  # task, data, model, num_params, metrics.*, setup.*
```

`setup.kwargs.combiner` and `setup.kwargs.basis` carry the rank, the number of
components and the basis of each run, so the ablation tables group directly by
them.
