"""Training losses and differentiable entropy estimates.

Loss functions follow ``loss_fn(batch, model) -> (loss, metrics)``: they run
the model on ``batch["input"]`` and return scalar loss and logging values.
"""

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["rate_distortion_loss", "sdf_loss", "soft_entropy", "laplace_entropy"]


def _iqr(x: Tensor) -> Tensor:
    """Return the interquartile range over all elements."""
    q1 = torch.quantile(x, 0.25)
    q3 = torch.quantile(x, 0.75)
    return q3 - q1


def soft_entropy(
    x: Tensor,
    num_bins: int | None = None,
    temperature: float = 1.0,
    eps: float = 1e-16,
) -> Tensor:
    """Estimate entropy in bits using differentiable soft bin assignments.

    Bin centers span the detached input range. Each value receives a softmax
    assignment based on squared distance; entropy is computed from the mean
    assignment. Gradients flow through assignments, not through the bin layout.

    Args:
        x: Input tensor; all elements are pooled.
        num_bins: Bin count. ``None`` uses the Freedman–Diaconis width
            ``2 * IQR(x) / N**(1/3)`` and rounds the count up.
        temperature: Assignment kernel width relative to bin width; larger is softer.
        eps: Constant added inside the logarithm.

    Returns:
        Scalar entropy tensor.
    """
    x_flat = x.reshape(-1, 1)

    x_min = x_flat.detach().min()
    x_max = x_flat.detach().max()

    if num_bins is None:
        iqr_value = _iqr(x_flat.detach())
        num_elements = x_flat.numel()
        bin_width = 2 * iqr_value / (num_elements ** (1 / 3))
        num_bins = int(torch.ceil((x_max - x_min) / bin_width).item())
    else:
        bin_width = (x_max - x_min) / num_bins

    bin_centers = torch.linspace(x_min, x_max, num_bins, dtype=x.dtype, device=x.device)

    squared_distances = (x_flat - bin_centers) ** 2
    sigma = bin_width * temperature
    logits = -squared_distances / (2 * sigma**2)
    assignments = F.softmax(logits, dim=1)

    probabilities = assignments.mean(dim=0)
    return -(probabilities * torch.log2(probabilities + eps)).sum()


def laplace_entropy(x: Tensor, eps: float = 1e-16) -> Tensor:
    """Return the differential entropy of a zero-mean Laplace fit, in bits.

    Uses the maximum-likelihood scale ``b = mean(abs(x))`` and entropy
    ``log2(2 * b * e + eps)``. All elements of ``x`` are pooled.
    """
    scale = x.abs().mean()
    return torch.log2(2 * scale * math.e + eps)


def rate_distortion_loss(
    batch: dict[str, Any],
    model: nn.Module,
    alpha: float = 0.0,
    eps: float = 1e-16,
) -> tuple[Tensor, dict[str, float]]:
    """Balance reconstruction MSE with estimated parameter entropy.

    The loss is ``MSE + alpha * entropy``, where entropy is the size-weighted
    mean Laplace entropy of trainable parameter tensors. It is evaluated only
    when ``alpha > 0``.

    Args:
        batch: Dictionary containing ``"input"`` and ``"target"`` tensors.
        model: Neural field to evaluate and whose parameters define the rate.
        alpha: Weight of the rate term.
        eps: Stability constant passed to :func:`laplace_entropy`.

    Returns:
        Scalar loss and float metrics: ``mse``, ``psnr`` (for targets in
        ``[-1, 1]``), ``loss``, and ``entropy`` when ``alpha > 0``.
    """
    output = model(batch["input"])
    target = batch["target"]
    metrics: dict[str, float] = {}

    mse = F.mse_loss(output, target)
    metrics["mse"] = mse.detach().item()
    metrics["psnr"] = 10 * math.log10(4.0 / metrics["mse"])

    if alpha > 0:
        entropy = 0
        total_params = 0
        for param in model.parameters():
            if not param.requires_grad:
                continue
            num_params = param.numel()
            total_params += num_params
            entropy += num_params * laplace_entropy(param, eps=eps)

        entropy /= total_params
        loss = mse + alpha * entropy
        metrics["entropy"] = entropy.detach().item()
    else:
        loss = mse

    metrics["loss"] = loss.detach().item()
    return loss, metrics


def sdf_loss(
    batch: dict[str, Any], model: nn.Module
) -> tuple[Tensor, dict[str, float]]:
    """Return L1 signed-distance loss and float metrics ``loss`` and ``l1``.

    Evaluates ``model(batch["input"])`` against ``batch["target"]``.
    """
    output = model(batch["input"])
    target = batch["target"]

    loss = F.l1_loss(output, target)
    value = loss.item()
    return loss, {"loss": value, "l1": value}
