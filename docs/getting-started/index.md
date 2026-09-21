# Getting started

Two pages take you from an empty environment to a trained neural field.

- [Installation](installation.md): requirements, the optional extras, and the benchmark data.
- [Your first neural field](quickstart.md): fit FUTON to an image, read the result, and swap in another model.

Everything is importable from the top-level package:

```python
import neurofield as nf

nf.FUTON, nf.SIREN, nf.InstantNGP        # models
nf.ImageCoordinateDataset                # datasets
nf.train, nf.evaluate, nf.chunked_inference
nf.psnr, nf.ssim, nf.lpips, nf.iou       # metrics
nf.compression, nf.nerf                  # image compression and radiance fields
```
