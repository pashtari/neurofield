"""Rendering and held-out-view evaluation for radiance fields."""

import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from ..metrics import lpips, psnr, ssim
from .datasets import BlenderDataset
from .renderers import VolumeRenderer

__all__ = ["render_image", "evaluate", "render_views"]


@contextmanager
def _eval_mode(field: nn.Module, renderer: VolumeRenderer) -> Iterator[None]:
    """Put both modules in eval mode, restoring training mode if ``field`` had it."""
    was_training = field.training
    field.eval()
    renderer.eval()
    try:
        yield
    finally:
        if was_training:
            field.train()
            renderer.train()


@torch.no_grad()
def render_image(
    field: nn.Module,
    renderer: VolumeRenderer,
    rays_o: Tensor,
    rays_d: Tensor,
    *,
    background: Tensor | None = None,
    chunk_size: int | None = None,
) -> dict[str, Tensor]:
    """Render rays in chunks without gradients or stratified jitter.

    Both modules enter eval mode and return to training mode if ``field``
    was training. Changing chunk size may change floating-point reductions.

    Args:
        field: Radiance field; rays move to its parameter device.
        renderer: Volume renderer, e.g. from :func:`create_renderer`.
        rays_o: Ray origins ``(*, 3)``, e.g. ``(H, W, 3)``.
        rays_d: Unit directions with the same shape as ``rays_o``.
        background: Shared background color ``(3,)``; ``None`` uses white.
        chunk_size: Rays per forward pass. ``None`` budgets ``2**19`` marching
            slots per pass, using the renderer's full marching step count.

    Returns:
        Dict with ``"rgb"`` ``(*, 3)``, ``"opacity"`` ``(*, 1)``, and
        ``"depth"`` ``(*, 1)``, preserving the leading ray dimensions.
    """
    if chunk_size is None:
        # Marching buffers are allocated before the per-ray sample cap is applied.
        chunk_size = max(2**19 // getattr(renderer, "num_marching_steps", 1024), 1)
    ray_shape = rays_o.shape[:-1]
    device = next(field.parameters()).device
    rays_o = rays_o.reshape(-1, 3).to(device)
    rays_d = rays_d.reshape(-1, 3).to(device)
    outputs: dict[str, list[Tensor]] = {"rgb": [], "opacity": [], "depth": []}
    with _eval_mode(field, renderer):
        for start in range(0, rays_o.shape[0], chunk_size):
            result = renderer(
                field,
                rays_o[start : start + chunk_size],
                rays_d[start : start + chunk_size],
                background=background,
            )
            for key in outputs:
                outputs[key].append(result[key])
    return {
        key: torch.cat(values).reshape(*ray_shape, -1)
        for key, values in outputs.items()
    }


@torch.no_grad()
def evaluate(
    field: nn.Module,
    renderer: VolumeRenderer,
    dataset: BlenderDataset,
    *,
    indices: Iterable[int] | None = None,
    metrics: Mapping[str, Callable[[Tensor, Tensor], float]] | None = None,
    chunk_size: int | None = None,
    return_images: bool = False,
) -> dict[str, Any]:
    """Evaluate held-out views against the dataset's background.

    Renders are clamped to ``[0, 1]`` before scoring. Both modules enter eval
    mode and return to training mode if ``field`` was training.

    Args:
        field: Radiance field; predictions and targets use its parameter device.
        renderer: Volume renderer, e.g. from :func:`create_renderer`.
        dataset: Posed RGBA images and the background used for compositing.
        indices: View indices; ``None`` evaluates all views.
        metrics: Mapping of names to functions ``fn(pred, target)`` on
            ``(3, H, W)`` images in ``[0, 1]``. ``None`` selects PSNR, SSIM, and
            LPIPS (AlexNet), the standard novel view synthesis metrics.
        chunk_size: Rays per forward pass; see :func:`render_image`.
        return_images: Include rendered images in the result.

    Returns:
        ``"views"`` contains one dict per view with its ``"view"`` index,
        ``"duration"`` in seconds (CUDA-synchronized), and metrics. ``"mean"``
        averages these entries except the index, or is empty for no views.
        When requested, ``"images"`` contains ``(H, W, 3)`` CPU tensors.
    """
    if metrics is None:
        metrics = {"psnr": psnr, "ssim": ssim, "lpips": lpips}
    indices = range(len(dataset)) if indices is None else indices
    device = next(field.parameters()).device
    background = dataset.background.to(device)

    views: list[dict[str, int | float]] = []
    images: list[Tensor] = []
    with _eval_mode(field, renderer):
        for index in indices:
            batch = dataset[index]
            start_time = time.time()
            rendered = render_image(
                field,
                renderer,
                batch["rays_o"],
                batch["rays_d"],
                background=background,
                chunk_size=chunk_size,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            duration = time.time() - start_time
            pred = rendered["rgb"].clamp(0.0, 1.0)
            target = dataset.composite(batch["rgba"].to(device), background)
            entry: dict[str, int | float] = {"view": int(index), "duration": duration}
            for name, metric in metrics.items():
                entry[name] = metric(pred.permute(2, 0, 1), target.permute(2, 0, 1))
            views.append(entry)
            if return_images:
                images.append(pred.cpu())

    keys = [key for key in views[0] if key != "view"] if views else []
    mean = {key: sum(view[key] for view in views) / len(views) for key in keys}
    result: dict[str, Any] = {"views": views, "mean": mean}
    if return_images:
        result["images"] = images
    return result


@torch.no_grad()
def render_views(
    field: nn.Module,
    renderer: VolumeRenderer,
    dataset: BlenderDataset,
    poses: Sequence[Tensor],
    *,
    chunk_size: int | None = None,
) -> list[NDArray[np.uint8]]:
    """Render an orbit (or any pose list) as uint8 frames for a video.

    Both modules enter eval mode and return to training mode if ``field``
    was training.

    Args:
        field: Radiance field.
        renderer: Volume renderer, e.g. from
            :func:`~neurofield.nerf.create_renderer`.
        dataset: Supplies the camera intrinsics and background color.
        poses: Camera-to-world matrices ``(4, 4)`` (e.g. from
            :func:`neurofield.nerf.pose_spherical`).
        chunk_size: Rays per forward pass; see :func:`render_image`.

    Returns:
        List of ``(H, W, 3)`` uint8 NumPy arrays, one per pose.
    """
    background = dataset.background.to(next(field.parameters()).device)
    frames: list[NDArray[np.uint8]] = []
    with _eval_mode(field, renderer):
        for pose in poses:
            rays_o, rays_d = dataset.rays_for_pose(pose)
            rendered = render_image(
                field,
                renderer,
                rays_o,
                rays_d,
                background=background,
                chunk_size=chunk_size,
            )
            rgb = (rendered["rgb"].clamp(0, 1) * 255).round().to(torch.uint8)
            frames.append(rgb.cpu().numpy())
    return frames
