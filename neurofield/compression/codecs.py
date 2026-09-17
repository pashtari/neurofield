"""Image codecs: PIL formats (JPEG, WebP, PNG, ...) and FUTON.

Each codec is an encoder that maps an image to bytes and a decoder that maps
the bytes back to an image tensor of shape :math:`(C, H, W)`, so any pair can be
benchmarked with :func:`~neurofield.compression.evaluate_compression`.
"""

import io
import os
from functools import partial
from typing import IO, Any

import torch
import torchvision.transforms.functional as FT
from PIL import Image
from torch import Tensor

from ..datasets import ImageCoordinateDataset, create_coordinates
from ..losses import rate_distortion_loss
from ..models import FUTON
from ..quantization import uniform_dequantize
from ..training import train
from ..utils import decode_dict, encode_dict

__all__ = ["encode_pil", "decode_pil", "encode_futon", "decode_futon"]


def encode_pil(
    image: str | os.PathLike[str] | IO[bytes] | Tensor, **kwargs: Any
) -> bytes:
    """Encode an image with a PIL codec.

    Args:
        image: Path or binary file object of an image file that PIL can open, or
            an image tensor of shape :math:`(C, H, W)` (``uint8``, or floating
            point in ``[0, 1]``), converted with
            :func:`torchvision.transforms.functional.to_pil_image`.
        **kwargs: Keyword arguments forwarded to :meth:`PIL.Image.Image.save`.
            Must include ``format`` (e.g. ``"JPEG"``, ``"WEBP"`` or ``"PNG"``)
            because PIL cannot infer it when saving to an in-memory buffer; may
            include codec options such as ``quality``.

    Returns:
        The encoded image file as bytes.

    Raises:
        ValueError: If ``image`` is not a ``str``, :class:`os.PathLike`,
            :class:`io.IOBase` or Tensor.

    Examples::

        >>> import torch
        >>> from neurofield.compression import decode_pil, encode_pil
        >>> image = torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8)
        >>> encoded = encode_pil(image, format="JPEG", quality=50)
        >>> decode_pil(encoded).shape
        torch.Size([3, 32, 32])
    """
    if isinstance(image, (str, os.PathLike, io.IOBase)):
        pil_image = Image.open(image)
    elif isinstance(image, Tensor):
        pil_image = FT.to_pil_image(image)
    else:
        raise ValueError("image must be a string, Path, IO[bytes] or Tensor")

    buffer = io.BytesIO()
    pil_image.save(buffer, **kwargs)
    return buffer.getvalue()


def decode_pil(encoded: bytes) -> Tensor:
    """Decode an image file stored in memory, e.g. the output of :func:`encode_pil`.

    Args:
        encoded: Bytes of an image file in any format PIL can read.

    Returns:
        The decoded image as a tensor of shape :math:`(C, H, W)` with the channels
        of the stored PIL mode (``uint8`` for 8-bit images).
    """
    return FT.pil_to_tensor(Image.open(io.BytesIO(encoded)))


def _check_module_spec(name: str, spec: Any) -> None:
    """Raise ``ValueError`` unless ``spec`` is a registry key or a (key, dict) pair."""
    is_key = isinstance(spec, str)
    is_pair = (
        isinstance(spec, (tuple, list))
        and len(spec) == 2
        and isinstance(spec[0], str)
        and isinstance(spec[1], dict)
    )
    if not (is_key or is_pair):
        raise ValueError(
            f"encode_futon requires `{name}` as a registry key or a (key, dict) "
            "pair, so that the bitstream can be loaded with "
            f"torch.load(weights_only=True); got {spec!r}"
        )


def encode_futon(
    image: str | os.PathLike[str] | Image.Image | Tensor,
    *,
    subsample: float = 1.0,
    lr: float = 0.1,
    num_epochs: int = 500,
    alpha: float = 0.005,
    quant_max: int = 127,
    quant_interval: int = -1,
    log_interval: int = 1,
    device: str | torch.device | None = None,
    seed: int = 0,
    **kwargs: Any,
) -> bytes:
    """Encode an image by fitting and serializing a quantized FUTON.

    Training uses :func:`~neurofield.losses.rate_distortion_loss` with weight
    ``alpha``. The bitstream contains quantized parameters, their scales, model
    config, image ID and spatial shape, serialized with ``torch.save`` and gzip.

    Args:
        image: Path or PIL image (converted to RGB), or ``uint8`` tensor
            of shape ``(C, H, W)``.
        subsample: Fraction of pixels sampled each epoch, in ``(0, 1]``.
        lr: Initial learning rate.
        num_epochs: Training epochs, each with one optimizer step.
        alpha: Weight of the parameters' Laplace entropy estimate in the loss.
        quant_max: Largest quantization level. Each parameter tensor is scaled
            by ``quant_max / max(abs(w))`` and rounded to integers.
        quant_interval: If positive, round parameters to the quantization grid
            in place every this many epochs. Final quantization always runs.
        log_interval: Epochs between log messages; non-positive disables them.
        device: Training device; ``None`` selects CUDA if available, else CPU.
        seed: Training seed, set after model construction. Controls pixel
            sampling but not parameter initialization.
        **kwargs: FUTON arguments, including required ``basis``, ``combiner``,
            and ``decoder`` registry keys or ``(key, params_dict)`` pairs.
            All config values must be plain Python or tensor types supported
            by ``torch.load(weights_only=True)``; callable specs cannot be
            decoded. Input/output widths default to 2 and the channel count.

    Returns:
        Compressed model and image metadata as bytes.

    Raises:
        ValueError: If ``basis``, ``combiner``, or ``decoder`` is missing or
            is not a registry key or ``(key, dict)`` pair.

    Example::

        encoded = encode_futon(
            image,
            basis=("cosine", {"num_components": 32}),
            combiner=("cp", {"rank": 32}),
            decoder="linear",
        )
        reconstruction = decode_futon(encoded)
    """
    for name in ("basis", "combiner", "decoder"):
        if name not in kwargs:
            raise ValueError(f"encode_futon requires the FUTON argument `{name}`")
        _check_module_spec(name, kwargs[name])

    dataset = ImageCoordinateDataset(image=image, subsample=subsample)
    model_config = {
        "in_features": dataset.input.shape[-1],
        "out_features": dataset.target.shape[-1],
        **kwargs,
    }
    model = FUTON(**model_config)
    train_results = train(
        model,
        dataset,
        loss_fn=partial(rate_distortion_loss, alpha=alpha),
        lr=lr,
        num_epochs=num_epochs,
        quantize=True,
        quant_max=quant_max,
        quant_interval=quant_interval,
        log_interval=log_interval,
        eval_interval=-1,
        device=device,
        seed=seed,
    )

    metadata = {
        "image": {"id": dataset.id, "shape": tuple(dataset.grid_size)},
        "model": {"config": model_config, "state": train_results["model_dict"]},
    }
    return encode_dict(metadata)


def decode_futon(encoded: bytes, *, device: str | torch.device | None = None) -> Tensor:
    """Reconstruct an image from :func:`encode_futon` bytes.

    Rebuilds the stored FUTON, dequantizes its parameters, and evaluates the
    full coordinate grid in one pass. Maps output from ``[-1, 1]`` to
    ``[0, 255]``, rounds, and casts to ``uint8`` without clamping.

    Args:
        encoded: Bitstream returned by :func:`encode_futon`.
        device: Decoding device; ``None`` leaves tensors on the default device.

    Returns:
        Reconstructed ``uint8`` image of shape ``(C, H, W)`` on ``device``.
    """
    metadata = decode_dict(encoded)
    model_config = metadata["model"]["config"]
    model_dict = metadata["model"]["state"]
    image_shape = metadata["image"]["shape"]

    coordinates = create_coordinates(image_shape).to(device)

    model = FUTON(**model_config)
    uniform_dequantize(model, model_dict)

    model.to(device)
    model.eval()
    with torch.no_grad():
        output = model(coordinates)

    return ImageCoordinateDataset.postprocess(output)
