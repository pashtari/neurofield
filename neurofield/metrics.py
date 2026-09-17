"""Evaluation metrics for images, volumes, and compressed representations.

Thin wrappers around torchmetrics that score a single sample and return a
Python float. Image metrics take channel-first ``(C, H, W)`` images, and their
``data_range`` defaults to the dtype convention of torchvision and
scikit-image: ``255`` for ``uint8`` images and ``1`` for floating-point images
in ``[0, 1]``. Pass it explicitly for other ranges, e.g.
``data_range=target.max()`` for MRI magnitudes.
"""

import math
from functools import cache
from typing import Any, Literal, TypeAlias

import torch
from torch import Tensor, nn
from torchmetrics.functional.classification import binary_jaccard_index
from torchmetrics.functional.image import (
    multiscale_structural_similarity_index_measure,
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)

__all__ = ["psnr", "ssim", "ms_ssim", "lpips", "nmse", "iou", "bits_per_pixel"]

_LPIPSNet: TypeAlias = Literal["alex", "vgg", "squeeze"]


def _data_range(target: Tensor, data_range: float | None) -> float:
    if data_range is not None:
        return float(data_range)
    return 1.0 if target.is_floating_point() else float(torch.iinfo(target.dtype).max)


def psnr(pred: Tensor, target: Tensor, data_range: float | None = None) -> float:
    """Peak signal-to-noise ratio in dB, pooling all elements."""
    return peak_signal_noise_ratio(
        pred.float(), target.float(), data_range=_data_range(target, data_range)
    ).item()


def ssim(
    pred: Tensor, target: Tensor, data_range: float | None = None, **kwargs: Any
) -> float:
    """Structural similarity of two ``(C, H, W)`` images (11x11 Gaussian window)."""
    return structural_similarity_index_measure(
        pred.float().unsqueeze(0),
        target.float().unsqueeze(0),
        data_range=_data_range(target, data_range),
        **kwargs,
    ).item()


def ms_ssim(
    pred: Tensor, target: Tensor, data_range: float | None = None, **kwargs: Any
) -> float:
    """Multi-scale structural similarity of two ``(C, H, W)`` images.

    The five default scales require both image sides to exceed 160 pixels.
    """
    return multiscale_structural_similarity_index_measure(
        pred.float().unsqueeze(0),
        target.float().unsqueeze(0),
        data_range=_data_range(target, data_range),
        **kwargs,
    ).item()


@cache
def _lpips_network(net: _LPIPSNet, device: torch.device) -> nn.Module:
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    return LearnedPerceptualImagePatchSimilarity(net, normalize=True).to(device).eval()


@torch.no_grad()
def lpips(
    pred: Tensor,
    target: Tensor,
    data_range: float | None = None,
    net: _LPIPSNet = "vgg",
) -> float:
    """Learned perceptual image patch similarity of two ``(3, H, W)`` images.

    Lower is better. The network is loaded once per backbone and device, and
    matches the reference ``lpips`` package. Inputs are clipped to the data range.

    References:
        Zhang et al., "The Unreasonable Effectiveness of Deep Features as a
        Perceptual Metric", CVPR 2018.
    """
    peak = _data_range(target, data_range)
    pred, target = ((x.float() / peak).clamp(0, 1).unsqueeze(0) for x in (pred, target))
    return _lpips_network(net, pred.device)(pred, target).item()


def nmse(pred: Tensor, target: Tensor) -> float:
    """Normalized MSE ``||pred - target||^2 / ||target||^2``, as in fastMRI."""
    pred, target = pred.float(), target.float()
    return ((pred - target).square().sum() / target.square().sum()).item()


def iou(pred: Tensor, target: Tensor) -> float:
    """Intersection over union of occupancies; values above zero are occupied."""
    return binary_jaccard_index((pred > 0).int(), (target > 0).int()).item()


def bits_per_pixel(encoded: bytes, image: Tensor) -> float:
    """Encoded size in bits per spatial pixel of a ``(C, *spatial)`` image."""
    return len(encoded) * 8 / math.prod(image.shape[1:])
