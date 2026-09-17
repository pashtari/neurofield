"""Neural field architectures, encodings, and tensor factorizations.

Public names are also available at the package top level, e.g. ``nf.FUTON``.
"""

from .dip import DIPSkip, DIPUNet
from .finer import FINER, FinerLayer
from .futon import (
    FUTON,
    ChebyshevBasis,
    CosineBasis,
    CPCombiner,
    HadamardCombiner,
    LanczosBasis,
    LegendreBasis,
    SincBasis,
    TRCombiner,
    TriangleBasis,
)
from .gaplanes import GAPlanes
from .gauss import Gauss, GaussLayer
from .grid_inr import GridINR
from .instant_ngp import HashEncoding, InstantNGP
from .mfn import MFN, GaborFilter
from .mlp import MLP, ReLULayer
from .multivector import FeatureGrid, MultiVector
from .pemlp import PEMLP, PositionalEncoding
from .rcs_matrix import RCSMatrix
from .rff import RFF, RFFEncoding
from .siren import SIREN, SineLayer
from .tensorf import TensoRF
from .wire import WIRE, ComplexGaborLayer, RealGaborLayer, RealWIRE

__all__ = [
    "MLP",
    "ReLULayer",
    "RFFEncoding",
    "RFF",
    "PositionalEncoding",
    "PEMLP",
    "SineLayer",
    "SIREN",
    "GaussLayer",
    "Gauss",
    "RealGaborLayer",
    "ComplexGaborLayer",
    "RealWIRE",
    "WIRE",
    "FinerLayer",
    "FINER",
    "GaborFilter",
    "MFN",
    "FeatureGrid",
    "MultiVector",
    "RCSMatrix",
    "CosineBasis",
    "SincBasis",
    "LegendreBasis",
    "ChebyshevBasis",
    "TriangleBasis",
    "LanczosBasis",
    "HadamardCombiner",
    "CPCombiner",
    "TRCombiner",
    "FUTON",
    "GAPlanes",
    "TensoRF",
    "HashEncoding",
    "InstantNGP",
    "GridINR",
    "DIPUNet",
    "DIPSkip",
]
