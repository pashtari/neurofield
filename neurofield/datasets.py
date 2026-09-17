"""Single-item datasets for fitting neural fields to images, volumes and MRI.

Coordinate inputs lie in ``[-1, 1]``. Each item contains ``id``, ``input``,
``target`` and ``_original``; the latter holds the reference signal for
metrics. All datasets have length one and ignore the item index.

Dataset methods follow ``load(path, ...)``, ``save(x, path, ...)``,
``preprocess(x)`` (raw signal to target), and ``postprocess(x)`` (model output
to reconstruction). DIP uses a fixed noise input instead of coordinates.
"""

import math
import os
import random
import warnings
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import functional as FT

__all__ = [
    "create_coordinates",
    "mask_to_boundary",
    "ImageCoordinateDataset",
    "MaskedImageCoordinateDataset",
    "OccupancyCoordinateDataset",
    "DIPImageDataset",
    "MRICoordinateDataset",
]


def create_coordinates(
    size: tuple[int, ...], domain: tuple[float, float] = (-1.0, 1.0), **kwargs: Any
) -> Tensor:
    """Return an evenly spaced coordinate grid of shape ``(*size, len(size))``.

    Each axis spans ``domain``, including both endpoints, with ``indexing="ij"``.
    Extra keyword arguments (e.g. ``dtype`` and ``device``) go to ``torch.linspace``.
    """
    axes = [torch.linspace(*domain, length, **kwargs) for length in size]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)


def _load_image(
    image: str | os.PathLike[str] | Image.Image | Tensor,
    mode: str,
    item_id: str | None,
) -> tuple[Tensor, str]:
    """Return a ``(C, H, W)`` tensor and item ID from a path, PIL image or tensor.

    Paths and PIL images are converted to ``mode``; tensors are used as is.
    The ID defaults to the path stem, or an empty string for in-memory images.
    """
    if isinstance(image, (str, os.PathLike)):
        with Image.open(image) as pil_image:
            return (
                FT.pil_to_tensor(pil_image.convert(mode)),
                item_id or Path(image).stem,
            )
    if isinstance(image, Image.Image):
        return FT.pil_to_tensor(image.convert(mode)), item_id or ""
    if isinstance(image, Tensor):
        return image, item_id or ""
    raise ValueError("image must be a str, os.PathLike, PIL.Image or Tensor")


def _save_png(image: Tensor, path: str | os.PathLike[str]) -> None:
    """Save a ``(C, H, W)`` image as PNG, appending ``.png`` if needed."""
    path = str(path)
    if not path.lower().endswith(".png"):
        path += ".png"
    FT.to_pil_image(image).save(path, format="PNG")


class ImageCoordinateDataset(Dataset):
    """An image as coordinate/pixel pairs, with optional random subsampling.

    Inputs have shape ``(H, W, 2)`` and targets ``(H, W, C)``, both in ``[-1, 1]``.
    ``_original`` is a uint8 image of shape ``(C, H, W)``. Subsampling flattens
    input and target to ``(N, 2)`` and ``(N, C)`` with shared random indices.

    Args:
        image: Path, PIL image, or uint8 tensor of shape ``(C, H, W)``.
        grid_size: Input grid size; ``None`` uses the image size. A different
            size supports super-resolution evaluation, but not subsampling.
        mode: PIL conversion mode for path and PIL inputs; ignored for tensors.
        max_samples: Optional cap on the number of sampled coordinates.
        subsample: Fraction in ``(0, 1]`` drawn on each access. The sample count
            is ``min(int(H * W * subsample), max_samples)`` when a cap is set.
        item_id: Item identifier; defaults to the path stem or an empty string.
        seed: Reseeds Python's global RNG. Subsampling uses the global PyTorch
            RNG and is unaffected by this seed.
    """

    def __init__(
        self,
        image: str | os.PathLike[str] | Image.Image | Tensor,
        grid_size: tuple[int, int] | None = None,
        mode: str = "RGB",
        max_samples: int | None = None,
        subsample: float = 1.0,
        item_id: str | None = None,
        seed: int = 0,
    ) -> None:
        if not 0 < subsample <= 1:
            raise ValueError(f"subsample must be in (0, 1], got {subsample}")
        if max_samples is not None and max_samples <= 0:
            raise ValueError(f"max_samples must be positive, got {max_samples}")

        self.original, self.id = _load_image(image, mode, item_id)
        self.grid_size = grid_size or self.original.shape[1:]
        self.input = self.create_input()
        self.target = self.preprocess(self.original)
        self.subsample = subsample
        self.max_samples = max_samples
        self.seed = seed
        random.seed(self.seed)

    # Tensors moved by ``to``; ``original`` intentionally stays on the CPU.
    _device_attrs: tuple[str, ...] = ("input", "target", "side", "mask", "boundary")

    def to(self, device: str | torch.device) -> Self:
        """Move training tensors to ``device`` in place and return this dataset.

        Includes side images, masks and boundaries when present, so subsampling
        runs on the training device. The reference ``original`` stays on CPU.
        """
        for name in self._device_attrs:
            value = getattr(self, name, None)
            if isinstance(value, Tensor):
                setattr(self, name, value.to(device))
        return self

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, Any]:
        data = {"id": self.id, "_original": self.original}
        if self.subsample < 1 or self.max_samples:
            num_coords = math.prod(self.grid_size)
            num_samples = int(num_coords * self.subsample)
            num_samples = min(num_samples, self.max_samples or num_samples)

            indices = torch.randperm(num_coords, device=self.input.device)[:num_samples]

            data["input"] = self.input.flatten(end_dim=-2)[indices, :]
            data["target"] = self.target.flatten(end_dim=-2)[indices, :]

        else:
            data["input"] = self.input
            data["target"] = self.target

        return data

    @staticmethod
    def load(path: str | os.PathLike[str], **kwargs: Any) -> Tensor:
        """Load an image as a ``(C, H, W)`` tensor.

        Keyword arguments go to PIL's ``convert`` (e.g. ``mode="RGB"``).
        Eight-bit modes produce uint8 tensors.
        """
        with Image.open(path) as image:
            return FT.pil_to_tensor(image.convert(**kwargs))

    @staticmethod
    def save(x: Tensor, path: str | os.PathLike[str]) -> None:
        """Save a ``(C, H, W)`` image as PNG, appending ``.png`` if needed."""
        _save_png(x, path)

    def create_input(self) -> Tensor:
        """Return coordinates in ``[-1, 1]`` with shape ``(*grid_size, len(grid_size))``."""
        return create_coordinates(self.grid_size, (-1.0, 1.0))

    @staticmethod
    def preprocess(x: Tensor) -> Tensor:
        """Map uint8 ``(C, H, W)`` pixels to float ``(H, W, C)`` values in ``[-1, 1]``."""
        x = x.float() * 2 / 255 - 1
        return x.movedim(0, -1)

    @staticmethod
    def postprocess(x: Tensor) -> Tensor:
        """Map float ``(H, W, C)`` predictions to uint8 ``(C, H, W)`` pixels.

        Predictions are clipped to ``[-1, 1]`` first, so out-of-range values
        saturate instead of wrapping around in the uint8 cast.
        """
        x = x.movedim(-1, 0).clamp(-1, 1)
        return ((x + 1) * 255 / 2).round().to(torch.uint8)


def mask_to_boundary(
    mask: Tensor,
    kernel_size: int = 3,
    mode: Literal["inner", "outer", "thick"] = "thick",
) -> Tensor:
    """Extract a binary mask boundary using dilation and erosion.

    Args:
        mask: Binary tensor of shape ``(H, W)``, ``(B, H, W)`` or ``(B, 1, H, W)``.
        kernel_size: Odd side length of the pooling neighborhood.
        mode: ``"inner"`` (mask minus erosion), ``"outer"`` (dilation minus mask),
            or ``"thick"`` (dilation minus erosion).

    Returns:
        Binary boundary with the same shape, dtype and device as ``mask``.
    """
    original_ndim = mask.ndim
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)
    elif mask.ndim == 4:
        if mask.shape[1] != 1:
            raise ValueError("mask should be single-channel (C=1) or 2/3-D tensor")
    else:
        raise ValueError("mask must be 2D, 3D (B,H,W), or 4D (B,1,H,W)")

    float_mask = mask.float()
    padding = kernel_size // 2

    dilation = F.max_pool2d(float_mask, kernel_size, stride=1, padding=padding)

    erosion = 1.0 - F.max_pool2d(
        1.0 - float_mask, kernel_size, stride=1, padding=padding
    )

    if mode == "inner":
        boundary = float_mask - erosion
    elif mode == "thick":
        boundary = dilation - erosion
    elif mode == "outer":
        boundary = dilation - float_mask
    else:
        raise ValueError("mode must be 'inner', 'outer' or 'thick'")

    boundary = (boundary > 0.0).to(mask.dtype)

    if original_ndim == 2:
        return boundary[0, 0]
    if original_ndim == 3:
        return boundary[:, 0]
    return boundary


class MaskedImageCoordinateDataset(ImageCoordinateDataset):
    """Coordinate/pixel pairs with an optional guide image and binary mask.

    The preprocessed guide is appended to coordinates as extra input channels.
    The mask and its thick boundary are returned for use by a custom loss.
    Inputs always use the image resolution.

    Args:
        image: Path, PIL image (converted to RGB), or uint8 ``(C, H, W)`` tensor.
        side: Optional guide: path, PIL image (converted to grayscale), or
            uint8 ``(C_s, H, W)`` tensor.
        mask: Optional mask: path, PIL image (converted to grayscale), or
            ``(1, H, W)`` / ``(H, W)`` tensor. Nonzero values are true.
        subsample: Fraction in ``(0, 1]`` drawn on each access.
        item_id: Item identifier; defaults to the path stem or an empty string.
        seed: Reseeds Python's global RNG, not the PyTorch RNG used for sampling.

    Items follow :class:`ImageCoordinateDataset`, with input shape
    ``(H, W, 2 + C_s)`` when a guide is present. Additional keys are ``side``
    (float ``(H, W, C_s)`` in ``[-1, 1]``), ``mask`` and ``boundary`` (bool
    ``(H, W)``), when provided. Subsampling flattens all training tensors
    using the same indices; ``_original`` remains the full image.
    """

    def __init__(
        self,
        image: str | os.PathLike[str] | Image.Image | Tensor,
        side: str | os.PathLike[str] | Image.Image | Tensor | None = None,
        mask: str | os.PathLike[str] | Image.Image | Tensor | None = None,
        subsample: float = 1.0,
        item_id: str | None = None,
        seed: int = 0,
    ) -> None:
        if not 0 < subsample <= 1:
            raise ValueError(f"subsample must be in (0, 1], got {subsample}")

        self.original, self.id = _load_image(image, "RGB", item_id)
        if side is not None:
            side, _ = _load_image(side, "L", None)
        if mask is not None:
            mask, _ = _load_image(mask, "L", None)

        self.grid_size = self.original.shape[-2:]
        self.input, self.side, self.mask, self.boundary = self.create_input(side, mask)
        self.target = self.preprocess(self.original)
        self.subsample = subsample
        self.seed = seed
        random.seed(self.seed)

    def create_input(
        self, side: Tensor | None, mask: Tensor | None
    ) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None]:
        """Return ``(input, side, mask, boundary)`` at the image resolution.

        ``side`` is a uint8 ``(C_s, H, W)`` image; ``mask`` has shape ``(1, H, W)``
        or ``(H, W)``. The guide is preprocessed and appended to coordinates.
        Mask and boundary become bool; absent auxiliary inputs return ``None``.
        """
        coords = create_coordinates(self.grid_size, (-1.0, 1.0))

        if isinstance(side, Tensor):
            side = self.preprocess(side)
            inputs = torch.cat([coords, side], dim=-1)
        else:
            side = None
            inputs = coords

        if isinstance(mask, Tensor):
            mask = mask.squeeze(0).bool()
            boundary = mask_to_boundary(mask, kernel_size=3, mode="thick")
        else:
            mask = None
            boundary = None

        return inputs, side, mask, boundary

    def __getitem__(self, index: int) -> dict[str, Any]:
        fields = {
            "input": self.input,
            "target": self.target,
            "side": self.side,
            "mask": self.mask,
            "boundary": self.boundary,
        }
        fields = {key: value for key, value in fields.items() if value is not None}
        if self.subsample < 1:
            num_coords = self.input.shape[0] * self.input.shape[1]
            num_samples = int(num_coords * self.subsample)
            indices = torch.randperm(num_coords, device=self.input.device)[:num_samples]
            # All fields share the (H, W) pixel grid in their leading dimensions.
            fields = {
                key: value.flatten(0, 1)[indices] for key, value in fields.items()
            }
        return {"id": self.id, "_original": self.original, **fields}


class OccupancyCoordinateDataset(ImageCoordinateDataset):
    """A voxelized shape as coordinate/occupancy pairs.

    Normalization and voxelization follow FINER/BACON. Items follow
    :class:`ImageCoordinateDataset`: inputs are ``(D, H, W, 3)`` coordinates,
    targets are ``(D, H, W, 1)`` floats in ``{-1, 1}``, and ``_original`` is a
    ``(1, D, H, W)`` float occupancy grid in ``{0, 1}``.

    Args:
        path: ``.xyz`` point cloud (columns ``x y z nx ny nz``) or a mesh
            readable by trimesh. Volumes are cached beside the source in
            ``.cache/<stem>_<resolution>.pt`` and reused even if the source changes.
        resolution: Samples per unit length after normalization into
            ``[-0.45, 0.45]^3``. Volumes are cropped to occupied bounds.
        grid_size: Coordinate grid size; ``None`` uses the volume size.
        max_samples: Optional cap on coordinates sampled per access.
        subsample: Fraction in ``(0, 1]`` drawn on each access.
        item_id: Item identifier; defaults to the path stem.
        seed: Reseeds Python's global RNG, not the PyTorch RNG used for sampling.

    References:
        Liu et al., "FINER", CVPR 2024.
        Lindell et al., "BACON", CVPR 2022.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        resolution: int = 256,
        grid_size: tuple[int, int, int] | None = None,
        max_samples: int | None = None,
        subsample: float = 1.0,
        item_id: str | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__(
            self.load(path, resolution),
            grid_size=grid_size,
            max_samples=max_samples,
            subsample=subsample,
            item_id=item_id or Path(path).stem,
            seed=seed,
        )

    def load(self, path: str | os.PathLike[str], resolution: int) -> Tensor:
        """Load or cache a float occupancy volume of shape ``(1, D, H, W)``.

        Point clouds use a KD-tree SDF; meshes use trimesh (``3d`` extra).
        The volume is cropped to occupied bounds. See the class documentation
        for normalization and cache naming.
        """
        path = Path(path)
        cache_path = path.parent / ".cache" / f"{path.stem}_{resolution}.pt"

        if cache_path.exists():
            return torch.load(cache_path, weights_only=True)

        if path.suffix.lower() == ".xyz":
            grid = self._voxelize_pointcloud(path, resolution)
        else:
            grid = self._voxelize_mesh(path, resolution)

        occupied = torch.where(grid > 0)
        grid = grid[
            occupied[0].min() : occupied[0].max() + 1,
            occupied[1].min() : occupied[1].max() + 1,
            occupied[2].min() : occupied[2].max() + 1,
        ].unsqueeze(0)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(grid, cache_path)
        return grid

    @staticmethod
    def _voxelize_pointcloud(path: Path, resolution: int) -> Tensor:
        """Approximate occupancy from ``x y z nx ny nz`` points using a KD-tree SDF.

        Center and globally scale points into ``[-0.45, 0.45]^3``. Each query
        uses the nearest point's offset projected onto the mean normal of its
        three nearest neighbors. Negative SDF values are occupied.
        """
        from scipy.spatial import cKDTree

        pointcloud = np.genfromtxt(path)
        vertices, normals = pointcloud[:, :3], pointcloud[:, 3:]

        norms = np.linalg.norm(normals, axis=-1, keepdims=True)
        norms[norms == 0] = 1.0
        normals = normals / norms

        # Leave a margin around the shape so its boundary stays inside the grid.
        vertices -= vertices.mean(axis=0)
        coord_max, coord_min = vertices.max(), vertices.min()
        vertices = (vertices - coord_min) / (coord_max - coord_min) * 0.9 - 0.45

        tree = cKDTree(vertices)

        axis = np.linspace(-0.5, 0.5, resolution)
        x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
        query_points = np.stack([x, y, z], axis=-1).reshape(-1, 3)

        # Bound memory while querying the three nearest surface normals.
        batch_size = 200_000
        sdf = np.empty(query_points.shape[0], dtype=np.float32)
        for i in range(0, len(query_points), batch_size):
            batch = query_points[i : i + batch_size]
            _, neighbor_indices = tree.query(batch, k=3)
            nearest_points = vertices[neighbor_indices[:, 0]]
            mean_normals = normals[neighbor_indices].mean(axis=1)
            mean_normals /= np.linalg.norm(mean_normals, axis=-1, keepdims=True) + 1e-10
            sdf[i : i + batch_size] = ((batch - nearest_points) * mean_normals).sum(
                axis=-1
            )

        occupancy = (
            (sdf < 0).astype(np.float32).reshape(resolution, resolution, resolution)
        )
        return torch.from_numpy(occupancy)

    @staticmethod
    def _voxelize_mesh(path: Path, resolution: int) -> Tensor:
        """Center and scale a mesh into ``[-0.45, 0.45]^3``, then voxelize and fill."""
        import trimesh

        mesh = trimesh.load(str(path), force="mesh")
        vertices = mesh.vertices.copy()
        vertices -= vertices.mean(axis=0)
        coord_max, coord_min = vertices.max(), vertices.min()
        vertices = (vertices - coord_min) / (coord_max - coord_min) * 0.9 - 0.45
        mesh.vertices = vertices

        voxels = mesh.voxelized(pitch=1.0 / resolution).fill()
        return torch.from_numpy(voxels.matrix.astype(np.float32))

    def save(
        self, x: Tensor, path: str | os.PathLike[str], smooth: bool = True
    ) -> None:
        """Export occupancy as a ``.dae`` mesh and ``.png`` render (requires ``3d``).

        ``x`` has shape ``(1, D, H, W)`` or ``(D, H, W)``; output suffixes replace
        that of ``path``. ``smooth=True`` applies Gaussian smoothing (sigma 1)
        before marching cubes at iso-level zero. Rendering needs a display, so
        without one only the mesh is written and a warning is issued.
        """
        import mcubes
        import open3d as o3d

        dae_path = Path(path).with_suffix(".dae")
        png_path = Path(path).with_suffix(".png")

        volume = x.squeeze(0).detach().cpu().numpy()
        if smooth:
            volume = mcubes.smooth(volume.copy(), method="gaussian", sigma=1)

        vertices, triangles = mcubes.marching_cubes(volume, 0)
        mcubes.export_mesh(vertices, triangles, dae_path)

        if not os.environ.get("DISPLAY"):
            # Open3D renders through GLFW, which aborts the process when it
            # cannot open a window, so check before creating one.
            warnings.warn(f"No display; saved {dae_path.name} without a render.")
            return

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.compute_vertex_normals()

        visualizer = o3d.visualization.Visualizer()
        visualizer.create_window(visible=False)
        visualizer.add_geometry(mesh)
        visualizer.update_renderer()
        visualizer.capture_screen_image(str(png_path), do_render=True)
        visualizer.destroy_window()

    @staticmethod
    def preprocess(x: Tensor) -> Tensor:
        """Map ``(C, D, H, W)`` occupancy to channels-last float targets in ``[-1, 1]``."""
        x = x.float() * 2 - 1
        return x.movedim(0, -1)

    @staticmethod
    def postprocess(x: Tensor) -> Tensor:
        """Threshold channels-last predictions at zero into ``(C, D, H, W)`` occupancy."""
        x = x.movedim(-1, 0)
        return (x > 0).float()


class DIPImageDataset(Dataset):
    """An image paired with fixed noise for Deep Image Prior (Ulyanov et al., 2018).

    Items contain ``id``, noise ``input``, float ``target`` in ``[0, 1]`` and
    uint8 ``_original``. Target and original both have shape ``(C, H, W)``.

    Args:
        image: Path, PIL image (converted to RGB), or uint8 ``(C, H, W)`` tensor.
        noise_shape: Input shape; ``None`` uses ``(noise_channels, H, W)``.
        noise_channels: Input channels when ``noise_shape`` is omitted.
        reg_noise_std: Gaussian noise added on each access using the global
            PyTorch RNG. Set to zero for evaluation.
        item_id: Item identifier; defaults to the path stem or an empty string.
        seed: Seed of a dedicated generator for fixed noise, drawn as
            ``0.1 * randn(noise_shape)``. Global RNGs are unaffected.
    """

    def __init__(
        self,
        image: str | os.PathLike[str] | Image.Image | Tensor,
        noise_shape: tuple[int, ...] | None = None,
        noise_channels: int = 16,
        reg_noise_std: float = 1 / 30,
        item_id: str | None = None,
        seed: int = 0,
    ) -> None:
        self.original, self.id = _load_image(image, "RGB", item_id)
        self.target = self.preprocess(self.original)

        noise_shape = noise_shape or (noise_channels, *self.original.shape[1:])
        generator = torch.Generator().manual_seed(seed)
        self.noise = torch.randn(noise_shape, generator=generator) * 0.1

        self.reg_noise_std = reg_noise_std

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, Any]:
        noise = self.noise
        if self.reg_noise_std > 0:
            noise = noise + torch.randn_like(noise) * self.reg_noise_std
        return {
            "id": self.id,
            "input": noise,
            "target": self.target,
            "_original": self.original,
        }

    @staticmethod
    def save(x: Tensor, path: str | os.PathLike[str]) -> None:
        """Save a ``(C, H, W)`` image as PNG, appending ``.png`` if needed."""
        _save_png(x, path)

    @staticmethod
    def preprocess(x: Tensor) -> Tensor:
        """Convert a uint8 image to float ``[0, 1]``, keeping its shape."""
        return x.float() / 255.0

    @staticmethod
    def postprocess(x: Tensor) -> Tensor:
        """Clamp to ``[0, 1]``, scale and round to uint8, keeping the image shape."""
        return (x.clamp(0, 1) * 255).round().to(torch.uint8)


class MRICoordinateDataset(Dataset):
    """Undersampled multi-coil MRI k-space for coordinate-based reconstruction.

    The network predicts interleaved real/imaginary coil images with
    ``num_coils * 2`` channels. A custom loss pads these to the k-space size,
    Fourier-transforms them and compares acquired lines against ``target``.

    Args:
        kspace: Complex ``(num_coils, H, W)`` tensor or fastMRI HDF5 path.
            File inputs also load ``reconstruction_rss`` when available.
        crop_size: Reconstruction field of view ``(crop_H, crop_W)``; defaults
            to the reference RSS image size. Required for tensor inputs.
        mask: Optional mask with ``W`` elements, reshaped to bool ``(1, 1, W)``.
        acceleration: Target acceleration when generating a mask.
        pattern: ``"uniform"`` or ``"random"`` mask generation.
        center_fraction: Fraction of center lines retained for calibration.
        slice_index: HDF5 slice to load; ``None`` selects the middle slice.
        item_id: Item identifier; defaults to the path stem or an empty string.
        seed: Mask generator seed.

    Item keys:
        ``id``: Item identifier.
        ``input``: ``(crop_H, crop_W, 2)`` coordinates in ``[-1, 1]``.
        ``target``: Complex ``(num_coils, H, W)`` masked, normalized k-space.
        ``mask``: Bool ``(1, 1, W)`` sampling mask.
        ``scale_factor``: Scalar normalization factor from :meth:`preprocess`.
        ``zero_filled``: ``(crop_H, crop_W)`` RSS reconstruction in original units.
        ``_original``: Reference RSS image of that shape, or ``None``.

    References:
        Zbontar et al., "fastMRI: An Open Dataset and Benchmarks for Accelerated
        MRI", arXiv 2018.
    """

    def __init__(
        self,
        kspace: str | os.PathLike[str] | Tensor,
        crop_size: tuple[int, int] | None = None,
        mask: Tensor | None = None,
        acceleration: float = 4.0,
        pattern: Literal["uniform", "random"] = "random",
        center_fraction: float = 0.08,
        slice_index: int | None = None,
        item_id: str | None = None,
        seed: int = 0,
    ) -> None:
        if isinstance(kspace, (str, os.PathLike)):
            self.id = item_id or Path(kspace).stem
            kspace, rss = self.load(kspace, slice_index=slice_index)
        elif isinstance(kspace, Tensor):
            self.id = item_id or ""
            rss = None
        else:
            raise ValueError("kspace must be a str, os.PathLike or Tensor")
        if kspace.ndim != 3 or not kspace.is_complex():
            raise ValueError(
                "kspace must be a complex (num_coils, H, W) tensor, got "
                f"{kspace.dtype} tensor of shape {tuple(kspace.shape)}"
            )

        self.kspace = kspace
        self.num_coils = kspace.shape[0]
        self.kspace_size = tuple(kspace.shape[-2:])
        if not crop_size and rss is None:
            raise ValueError(
                "crop_size must be given when no reconstruction_rss is available "
                "(e.g. for k-space tensor inputs)"
            )
        self.crop_size = crop_size or tuple(rss.shape[-2:])

        if mask is None:
            mask = self.create_undersampling_mask(
                num_cols=kspace.shape[-1],
                acceleration=acceleration,
                pattern=pattern,
                center_fraction=center_fraction,
                seed=seed,
            )
        # (1, 1, W), broadcasts against (num_coils, H, W) k-space
        self.mask = mask.reshape(1, 1, -1).bool()

        kspace_normalized, self.scale_factor = self.preprocess(
            self.kspace, self.mask, self.crop_size
        )

        self.input = create_coordinates(self.crop_size, (-1.0, 1.0))
        self.target = kspace_normalized * self.mask
        self.original = rss

        self.zero_filled = self.zero_filled_reconstruction(
            kspace, self.mask, self.crop_size
        )

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "id": self.id,
            "input": self.input,
            "target": self.target,
            "mask": self.mask,
            "scale_factor": self.scale_factor,
            "zero_filled": self.zero_filled,
            "_original": self.original,
        }

    @staticmethod
    def load(
        path: str | os.PathLike[str], slice_index: int | None = None
    ) -> tuple[Tensor, Tensor | None]:
        """Load one fastMRI slice as ``(kspace, rss)`` (requires h5py).

        K-space has shape ``(num_coils, H, W)``; the reference RSS image has shape
        ``(crop_H, crop_W)`` or is ``None`` if absent. ``slice_index=None`` selects
        the middle slice.
        """
        import h5py

        with h5py.File(path, "r") as data_file:
            kspace_all = torch.from_numpy(data_file["kspace"][()])
            if slice_index is None:
                slice_index = kspace_all.shape[0] // 2
            rss_target = None
            if "reconstruction_rss" in data_file:
                rss_target = torch.from_numpy(
                    data_file["reconstruction_rss"][slice_index]
                )

        return kspace_all[slice_index], rss_target

    @staticmethod
    def save(x: Tensor, path: str | os.PathLike[str]) -> None:
        """Min-max normalize a ``(H, W)`` magnitude image and save an 8-bit PNG."""
        image = x.detach().cpu().float()
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        _save_png((image * 255).round().to(torch.uint8).unsqueeze(0), path)

    @staticmethod
    def create_undersampling_mask(
        num_cols: int,
        acceleration: float,
        pattern: Literal["uniform", "random"] = "random",
        center_fraction: float = 0.08,
        seed: int = 0,
    ) -> Tensor:
        """Return a bool Cartesian sampling mask of shape ``(num_cols,)``.

        Always keeps ``round(num_cols * center_fraction)`` center lines.
        ``"random"`` adds lines to reach ``round(num_cols / acceleration)`` total,
        unless the center alone exceeds this budget. ``"uniform"`` adds every
        ``round(acceleration)``-th line from a random offset, without subtracting
        center lines from the budget. A dedicated generator uses ``seed``.
        """
        generator = torch.Generator().manual_seed(seed)

        mask = torch.zeros(num_cols, dtype=torch.bool)

        num_center = round(num_cols * center_fraction)
        center_start = (num_cols - num_center) // 2
        mask[center_start : center_start + num_center] = True

        if pattern == "uniform":
            step = round(acceleration)
            offset = torch.randint(0, step, (1,), generator=generator).item()
            mask[offset::step] = True
        elif pattern == "random":
            num_outer = max(round(num_cols / acceleration) - num_center, 0)
            outer_indices = torch.where(~mask)[0]
            permutation = torch.randperm(len(outer_indices), generator=generator)[
                :num_outer
            ]
            mask[outer_indices[permutation]] = True
        else:
            raise ValueError(f"Unknown pattern '{pattern}'. Use 'uniform' or 'random'.")

        return mask

    @staticmethod
    def zero_filled_reconstruction(
        kspace: Tensor, mask: Tensor, crop_size: tuple[int, int] | None = None
    ) -> Tensor:
        """Return the RSS magnitude of masked k-space, optionally center-cropped.

        ``kspace`` is complex ``(num_coils, H, W)`` and ``mask`` must broadcast to
        it. The orthonormal inverse FFT uses ``ifftshift`` before and after.
        Returns ``(crop_H, crop_W)`` with ``crop_size``, otherwise ``(H, W)``.
        """
        kspace_masked = kspace * mask

        coil_images = torch.fft.ifftshift(
            torch.fft.ifft2(
                torch.fft.ifftshift(kspace_masked, dim=(-2, -1)), norm="ortho"
            ),
            dim=(-2, -1),
        )

        if crop_size is not None:
            height, width = coil_images.shape[-2:]
            crop_height, crop_width = crop_size
            coil_images = coil_images[
                ...,
                (height - crop_height) // 2 : (height + crop_height) // 2,
                (width - crop_width) // 2 : (width + crop_width) // 2,
            ]

        return (coil_images.real**2 + coil_images.imag**2).sum(dim=-3).sqrt()

    def preprocess(
        self, x: Tensor, mask: Tensor, crop_size: tuple[int, int], eps: float = 1e-16
    ) -> tuple[Tensor, Tensor]:
        """Return normalized k-space and its scalar scale factor.

        Divides complex ``(num_coils, H, W)`` k-space by the maximum of its
        masked, cropped zero-filled RSS image, clamped to ``eps``. Linearity
        of the FFT makes the same scale apply in both domains.
        """
        zero_filled = self.zero_filled_reconstruction(x, mask, crop_size)
        scale_factor = torch.clamp(zero_filled.max(), min=eps)
        return x / scale_factor, scale_factor

    def postprocess(self, x: Tensor) -> Tensor:
        """Convert interleaved coil predictions to RSS magnitude in original units.

        ``x`` has shape ``(crop_H, crop_W, num_coils * 2)``. Its norm over the
        last dimension gives RSS directly, without constructing complex tensors.
        The magnitude is clamped to ``[0, 1]`` then multiplied by ``scale_factor``.
        """
        return torch.clamp(torch.norm(x, dim=-1), min=0.0, max=1.0) * self.scale_factor
