"""Posed RGBA images and ray sampling for NeRF synthetic (Blender) scenes.

Camera defaults follow torch-ngp's Blender recipe used by FINER:
``scene_scale=0.8``, ``bound=1``, and ``near=0.2``. Images load at full
resolution by default; ``downsample=4, skip=4`` gives FINER's 25 training
views at 200 x 200. Straight RGBA keeps the background a rendering choice.
"""

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Self

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from .cameras import focal_from_fov, generate_ray_directions, generate_rays

__all__ = ["BlenderDataset"]


class BlenderDataset(Dataset[dict[str, Tensor]]):
    """A Blender scene with full-view access and random ray sampling.

    ``dataset[i]`` returns full-image rays and RGBA. :meth:`sample_rays` draws
    uniformly across views and pixels, generating rays from stored poses to
    avoid a full ray buffer. Poses retain Blender's coordinate convention and
    have their translations multiplied by ``scene_scale``.

    Args:
        root: Scene directory containing ``transforms_{split}.json`` and images.
        split: Split name, usually ``"train"``, ``"val"``, or ``"test"``.
        downsample: Integer resolution reduction factor, using area averaging.
            ``4`` gives 200 x 200 Blender images; for divisible sizes this
            matches torch-ngp's area filter up to uint8 rounding.
        skip: Keep every ``skip``-th frame in file order. ``4`` selects 25 of
            the 100 training views; use ``1`` for evaluation.
        scene_scale: Scale applied to camera positions.
        bound: Half-extent of the scene box ``[-bound, bound]^3``.
        near: Near-plane floor, stored for constructing the renderer.
        background: Scalar or three-channel background in ``[0, 1]``.
            White is the standard Blender evaluation background.

    Attributes:
        images: ``(N, H, W, 4)`` float32 straight RGBA in ``[0, 1]``.
        poses: ``(N, 4, 4)`` float32 camera-to-world matrices, with scaled positions.
        directions: Shared ``(H, W, 3)`` camera-frame ray directions.
        radius: Mean camera distance from the origin after scaling.
        focal: Focal length in pixels at the loaded resolution.
        height: Loaded image height in pixels.
        width: Loaded image width in pixels.
        far: ``None``: each ray ends at the scene-box exit.
        aabb: ``(6,)`` float32 scene box ``(-bound, ..., bound)``.
        background: ``(3,)`` float32 background color.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        split: str = "train",
        downsample: int = 1,
        skip: int = 1,
        scene_scale: float = 0.8,
        bound: float = 1.0,
        near: float = 0.2,
        background: float | Sequence[float] | Tensor = 1.0,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.downsample = int(downsample)
        self.skip = int(skip)
        self.scene_scale = float(scene_scale)
        self.bound = float(bound)
        self.near = float(near)
        self.far: float | None = None
        self.aabb = torch.tensor((-bound,) * 3 + (bound,) * 3, dtype=torch.float32)
        self.background = torch.as_tensor(background, dtype=torch.float32).expand(3)

        with open(self.root / f"transforms_{split}.json") as file:
            metadata = json.load(file)

        images: list[Tensor] = []
        poses: list[Tensor] = []
        for frame in metadata["frames"][:: self.skip]:
            path = self.root / f"{frame['file_path']}.png"
            with Image.open(path) as image:
                rgba = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)
            if rgba.shape[-1] == 3:
                rgba = torch.cat([rgba, torch.ones_like(rgba[..., :1])], dim=-1)
            elif rgba.shape[-1] != 4:
                raise ValueError(
                    f"{path}: expected an RGB or RGBA image, got "
                    f"{rgba.shape[-1]} channel(s)"
                )
            if self.downsample > 1:
                # Average straight RGBA to match the Blender training protocol.
                rgba = (
                    F.interpolate(
                        rgba.permute(2, 0, 1).unsqueeze(0),
                        size=(
                            rgba.shape[0] // self.downsample,
                            rgba.shape[1] // self.downsample,
                        ),
                        mode="area",
                    )
                    .squeeze(0)
                    .permute(1, 2, 0)
                )
            images.append(rgba)
            pose = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
            pose[:3, 3] *= self.scene_scale
            poses.append(pose)
        self.images = torch.stack(images)
        self.poses = torch.stack(poses)

        self.radius = self.poses[:, :3, 3].norm(dim=-1).mean().item()
        self.height, self.width = self.images.shape[1:3]
        self.focal = focal_from_fov(float(metadata["camera_angle_x"]), self.width)
        self.directions = generate_ray_directions(self.height, self.width, self.focal)

    def to(self, device: str | torch.device) -> Self:
        """Move stored tensors to ``device`` in place and return this dataset.

        Subsequent sampling runs on that device, avoiding per-step transfers.
        Unlike ``nn.Module.to``, this method accepts only a device.
        """
        for name in ("images", "poses", "directions", "aabb", "background"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return full-image rays and RGBA pixels for one view.

        ``"rays_o"`` and unit ``"rays_d"`` have shape ``(H, W, 3)``;
        ``"rgba"`` contains straight RGBA in ``[0, 1]`` of shape ``(H, W, 4)``.
        """
        rays_o, rays_d = generate_rays(self.poses[index], self.directions)
        return {"rays_o": rays_o, "rays_d": rays_d, "rgba": self.images[index]}

    def composite(self, rgba: Tensor, background: Tensor | None = None) -> Tensor:
        """Composite straight RGBA onto a background color.

        Args:
            rgba: Straight (non-premultiplied) RGBA in ``[0, 1]`` of shape
                ``(*, 4)``.
            background: Color to composite onto, broadcastable to the RGB
                shape ``(*, 3)``. ``None`` uses :attr:`background`.

        Returns:
            Composited RGB of shape ``(*, 3)``.
        """
        background = self.background if background is None else background
        rgb, alpha = rgba[..., :3], rgba[..., 3:]
        return rgb * alpha + background * (1.0 - alpha)

    def rays_for_pose(self, c2w: Tensor) -> tuple[Tensor, Tensor]:
        """Generate full-image rays for an arbitrary camera pose.

        Args:
            c2w: Camera-to-world pose of shape ``(4, 4)`` or ``(3, 4)`` (e.g.
                from :func:`~neurofield.nerf.pose_spherical`); it is moved to
                the device of :attr:`directions`.

        Returns:
            ``(rays_o, rays_d)``, each of shape ``(H, W, 3)``, with ``rays_d``
            unit norm.
        """
        return generate_rays(c2w.to(self.directions.device), self.directions)

    def sample_rays(
        self, num_rays: int, generator: torch.Generator | None = None
    ) -> dict[str, Tensor]:
        """Sample a random batch of rays uniformly over all views and pixels.

        Args:
            num_rays: Number of rays to sample.
            generator: Random number generator. Indices are drawn on the
                generator's device and then moved to the storage device, so a
                CPU generator also works with CUDA storage. ``None`` draws from
                the global RNG on the storage device.

        Returns:
            Dict with ``"rays_o"`` and ``"rays_d"`` of shape ``(num_rays, 3)``
            (``rays_d`` unit norm) and ``"rgba"`` of shape ``(num_rays, 4)``
            (straight RGBA), on the device of :attr:`images`.
        """
        device = self.images.device
        num_pixels = self.height * self.width
        # Draw on the generator's device to support CPU generators with CUDA data.
        generator_device = generator.device if generator is not None else device
        view_indices = torch.randint(
            len(self.images), (num_rays,), device=generator_device, generator=generator
        ).to(device)
        pixel_indices = torch.randint(
            num_pixels, (num_rays,), device=generator_device, generator=generator
        ).to(device)
        directions = self.directions.reshape(-1, 3)[pixel_indices]
        rays_o, rays_d = generate_rays(self.poses[view_indices], directions)
        rgba = self.images.reshape(len(self.images), -1, 4)[view_indices, pixel_indices]
        return {"rays_o": rays_o, "rays_d": rays_d, "rgba": rgba}
