"""Symmetric per-tensor quantization for training and compression."""

import torch
from torch import Tensor, nn

__all__ = ["uniform_quantize", "uniform_dequantize"]


@torch.no_grad()
def uniform_quantize(
    model: nn.Module, quant_max: int
) -> tuple[nn.Module, dict[str, dict[str, Tensor]]]:
    """Fake-quantize model parameters in place and return their integer codes.

    Each parameter becomes ``clamp(round(s * w), -quant_max, quant_max) / s``,
    where ``s = quant_max / max(abs(w))``. Pass a model copy to retain the
    original weights. All-zero tensors become NaN; complex parameters are
    unsupported.

    Args:
        model: Module to quantize in place.
        quant_max: Largest integer level, less than ``2**63``. Codes use the
            smallest signed integer dtype that can hold this value.

    Returns:
        The same model and a dictionary mapping parameter names to
        ``{"quant": codes, "scale": s}``, both detached CPU tensors.
    """
    for dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        if quant_max <= torch.iinfo(dtype).max:
            break
    else:
        raise ValueError(f"quant_max must be less than 2**63, got {quant_max}")

    model_dict: dict[str, dict[str, Tensor]] = {}
    for name, param in model.named_parameters():
        scale = quant_max / param.abs().max()
        codes = torch.clamp(torch.round(scale * param), -quant_max, quant_max)
        param.copy_(codes / scale)

        model_dict[name] = {
            "quant": codes.cpu().to(dtype),
            "scale": scale.cpu(),
        }

    return model, model_dict


@torch.no_grad()
def uniform_dequantize(
    model: nn.Module, model_dict: dict[str, dict[str, Tensor]]
) -> nn.Module:
    """Load parameters from :func:`uniform_quantize` codes in place.

    ``model_dict`` must have the same parameter names as ``model``. Each
    parameter is overwritten with ``quant.float() / scale.float()``.
    Returns the same model.
    """
    for name, param in model.named_parameters():
        codes = model_dict[name]["quant"].float()
        scale = model_dict[name]["scale"].float()
        param.copy_(codes / scale)

    return model
