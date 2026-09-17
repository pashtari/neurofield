"""NeuroField: neural fields in PyTorch.

Models, datasets, training, evaluation and utilities are available directly
as ``nf.FUTON``, ``nf.train``, etc. Compression and novel view synthesis live
in ``nf.compression`` and ``nf.nerf``.

Example::

    import neurofield as nf

    model = nf.FUTON(
        in_features=2,
        out_features=3,
        basis=("cosine", {"num_components": 64}),
        combiner=("cp", {"rank": 32}),
        decoder="linear",
    )
"""

__version__ = "0.1.0"

from . import (
    compression,
    datasets,
    evaluation,
    losses,
    metrics,
    models,
    nerf,
    quantization,
    training,
    utils,
    visualization,
)
from .datasets import *
from .evaluation import *
from .losses import *
from .metrics import *
from .models import *
from .quantization import *
from .training import *
from .utils import *
from .visualization import *

__all__ = [
    *datasets.__all__,
    *losses.__all__,
    *metrics.__all__,
    *models.__all__,
    *quantization.__all__,
    *utils.__all__,
    *visualization.__all__,
    *training.__all__,
    *evaluation.__all__,
    "compression",
    "nerf",
]
