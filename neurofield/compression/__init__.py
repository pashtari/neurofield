"""Image codecs, rate-distortion evaluation, and integer arithmetic coding.

PIL and FUTON codecs map image tensors to bytes and back. Benchmark either
with :func:`evaluate_compression`. These APIs live under ``nf.compression``
rather than the top-level package namespace.
"""

from . import arithmetic_coding, codecs, evaluation
from .arithmetic_coding import *
from .codecs import *
from .evaluation import *

__all__ = [
    *codecs.__all__,
    *evaluation.__all__,
    *arithmetic_coding.__all__,
]
