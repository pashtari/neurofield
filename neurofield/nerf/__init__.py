"""Novel view synthesis with radiance fields and volume renderers.

Use any pointwise neural field as the density network in :class:`RadianceField`.
:func:`create_renderer` selects nerfacc when its CUDA extension works, or a
PyTorch implementation otherwise. Training defaults follow torch-ngp's Blender
recipe: scaled cameras, random backgrounds, uniform ray marching, and Adam
with exponential learning-rate decay.

``nf.nerf.train`` and ``nf.nerf.evaluate`` fit and evaluate posed images.
The package-level functions fit coordinate-value pairs directly.

Example::

    import neurofield as nf

    dataset = nf.nerf.BlenderDataset(
        "data/nerf/blender/lego", downsample=4, skip=4
    )
    field = nf.nerf.RadianceField(
        nf.InstantNGP,
        ("mlp", {"hidden_features": 64, "hidden_layers": 2}),
        aabb=dataset.aabb,
    )
    renderer = nf.nerf.create_renderer(aabb=dataset.aabb, near=dataset.near)
    result = nf.nerf.train(field, renderer, dataset, num_steps=20000)
    test_set = nf.nerf.BlenderDataset(
        "data/nerf/blender/lego", split="test", downsample=4
    )
    scores = nf.nerf.evaluate(field, renderer, test_set)  # PSNR, SSIM, LPIPS
"""

from . import cameras, datasets, evaluation, fields, renderers, training
from .cameras import *
from .datasets import *
from .evaluation import *
from .fields import *
from .renderers import *
from .training import *

__all__ = [
    *cameras.__all__,
    *datasets.__all__,
    *fields.__all__,
    *renderers.__all__,
    *training.__all__,
    *evaluation.__all__,
]
