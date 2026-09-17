"""Occupancy-grid volume rendering with PyTorch and nerfacc backends.

:func:`create_renderer` selects nerfacc when its CUDA extension works, or
PyTorch otherwise. Both use uniform marching and emission-absorption
compositing. Ray directions must be unit vectors so near/far bounds, step
size, and depth are measured in world-space distance.

Call ``update_occupancy(field, step)`` before each training step. Defaults
follow torch-ngp's Blender recipe: a unit box, near plane 0.2, step size
``sqrt(3) * extent / 1024``, and a 128-cubed density grid thresholded at
``min(mean density, 10)``.
"""

import math
from collections.abc import Sequence
from typing import Any, Literal

import torch
from torch import Tensor, nn

__all__ = [
    "is_nerfacc_available",
    "create_renderer",
    "VolumeRenderer",
    "PyTorchRenderer",
    "NerfAccRenderer",
]

try:
    import nerfacc
except Exception:
    # Broken binary extensions can raise more than ImportError.
    nerfacc = None


def is_nerfacc_available() -> bool:
    """Check whether nerfacc imports and its CUDA intersection kernel works.

    Some installations import successfully but lack usable CUDA kernels, so
    availability is checked with a small ``ray_aabb_intersect`` call.
    """
    if nerfacc is None or not torch.cuda.is_available():
        return False
    try:
        rays_o = torch.zeros(1, 3, device="cuda")
        rays_d = torch.tensor([[0.0, 0.0, 1.0]], device="cuda")
        aabb = torch.tensor([[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]], device="cuda")
        nerfacc.ray_aabb_intersect(rays_o, rays_d, aabb)
        return True
    except Exception:
        return False


def create_renderer(
    backend: Literal["auto", "pytorch", "nerfacc"] = "auto", **kwargs: Any
) -> "VolumeRenderer":
    """Create a renderer, selecting a working CUDA backend when available.

    Args:
        backend: ``"nerfacc"`` (:class:`NerfAccRenderer`), ``"pytorch"``
            (:class:`PyTorchRenderer`), or ``"auto"`` (nerfacc when
            :func:`is_nerfacc_available`, else pure PyTorch).
        **kwargs: Forwarded to the renderer constructor (``aabb``, ``near``,
            ``far``, ``render_step_size``, ...; see :class:`VolumeRenderer`).
    """
    if backend == "auto":
        backend = "nerfacc" if is_nerfacc_available() else "pytorch"
    if backend == "nerfacc":
        return NerfAccRenderer(**kwargs)
    if backend == "pytorch":
        return PyTorchRenderer(**kwargs)
    raise ValueError(
        f"Unknown backend {backend!r}; use 'auto', 'nerfacc', or 'pytorch'"
    )


class VolumeRenderer(nn.Module):
    """Shared configuration and interface for occupancy-grid volume renderers.

    Subclasses implement :meth:`update_occupancy` and :meth:`forward`.
    Construct a concrete backend through :func:`create_renderer`.

    Args:
        aabb: Scene box ``(xmin, ymin, zmin, xmax, ymax, zmax)``; samples lie
            inside it. Stored as a non-persistent float32 buffer.
        near: Near-plane floor. Marching starts at the box entry when further.
        far: Far-plane cap; ``None`` uses the box exit and is stored as infinity.
        render_step_size: Marching step. ``None`` uses
            ``sqrt(3) * max_extent / 1024``.
        max_samples_per_ray: Keep at most this many occupied samples per ray,
            front to back; ``None`` removes the cap. Truncated tails become
            background. This bounds field queries but does not reduce the
            PyTorch backend's marching buffers.
        grid_resolution: Occupancy-grid cells per axis.
        update_interval: Training steps between occupancy updates.
        density_threshold: A cell is occupied when its density exceeds
            ``min(grid mean, density_threshold)``.
        ema_decay: Decay in the per-cell update ``max(decay * density, sigma)``.
        warmup_steps: Update every cell during warmup. Afterwards, sample
            ``num_cells // 4`` uniform cells and up to that many occupied cells.

    Attributes:
        num_marching_steps: ``ceil(box diagonal / render_step_size)``, at least 1.
            With unit directions, this covers the longest possible box crossing.
    """

    def __init__(
        self,
        aabb: Tensor | Sequence[float] = (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
        near: float = 0.2,
        far: float | None = None,
        render_step_size: float | None = None,
        max_samples_per_ray: int | None = 1024,
        grid_resolution: int = 128,
        update_interval: int = 16,
        density_threshold: float = 10.0,
        ema_decay: float = 0.95,
        warmup_steps: int = 256,
    ) -> None:
        super().__init__()
        aabb = torch.as_tensor(aabb, dtype=torch.float32)
        self.register_buffer("aabb", aabb, persistent=False)
        self.near = float(near)
        self.far = math.inf if far is None else float(far)
        if render_step_size is None:
            max_extent = (aabb[3:] - aabb[:3]).max().item()
            render_step_size = math.sqrt(3.0) * max_extent / 1024.0
        self.render_step_size = float(render_step_size)
        # The box diagonal bounds every ray's path through the scene.
        diagonal = (aabb[3:] - aabb[:3]).norm().item()
        self.num_marching_steps = max(math.ceil(diagonal / self.render_step_size), 1)
        self.max_samples_per_ray = (
            None if max_samples_per_ray is None else int(max_samples_per_ray)
        )
        self.grid_resolution = int(grid_resolution)
        self.update_interval = int(update_interval)
        self.density_threshold = float(density_threshold)
        self.ema_decay = float(ema_decay)
        self.warmup_steps = int(warmup_steps)

    def update_occupancy(self, field: nn.Module, step: int) -> None:
        """Update the occupancy grid from the field's current density.

        Call once per training step, before rendering.

        Args:
            field: Radiance field exposing ``query_density(x)``, which maps
                world coordinates ``(N, 3)`` to densities ``(N,)``.
            step: 0-based training step. The grid is updated only when
                ``step % update_interval == 0``; every cell is re-evaluated
                while ``step < warmup_steps``.
        """
        raise NotImplementedError

    def forward(
        self,
        field: nn.Module,
        rays_o: Tensor,
        rays_d: Tensor,
        background: Tensor | None = None,
    ) -> dict[str, Tensor | int]:
        """Render a batch of rays through a radiance field.

        Args:
            field: Radiance field; ``field(x, directions)`` returns
                ``(rgb, sigma)`` of shapes ``(N, 3)`` and ``(N,)`` for world
                coordinates and view directions ``(N, 3)``, and
                ``field.query_density(x)`` returns ``(N,)``.
            rays_o: Ray origins ``(R, 3)``.
            rays_d: Unit-norm ray directions ``(R, 3)``.
            background: Background color composited behind each ray, either
                shared ``(3,)`` or per ray ``(R, 3)``. ``None`` selects white.

        Returns:
            Dict with ``"rgb"`` ``(R, 3)``, ``"opacity"`` ``(R, 1)`` (summed
            compositing weights), ``"depth"`` ``(R, 1)`` (expected termination
            distance along the ray, conditioned on a hit) and
            ``"num_samples"`` (``int``, the number of samples composited).
        """
        raise NotImplementedError


class PyTorchRenderer(VolumeRenderer):
    """Occupancy-grid volume renderer implemented entirely in PyTorch.

    Marches each ray from box entry to exit at a fixed step, queries the field
    at occupied samples, and composites emission-absorption weights. Skipped
    samples are assigned zero density and leave transmittance unchanged.
    The grid starts fully occupied until the first :meth:`update_occupancy`.

    Constructor arguments are documented in :class:`VolumeRenderer`.

    Attributes:
        density_grid: ``(G, G, G)`` running cell densities, where
            ``G = grid_resolution``. Non-persistent buffer.
        binaries: Boolean occupancy mask of the same shape, named to match
            nerfacc's estimator. Non-persistent buffer.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        resolution = self.grid_resolution
        # Rebuild occupancy for the scene box instead of restoring a stale grid.
        self.register_buffer(
            "density_grid",
            torch.zeros(resolution, resolution, resolution),
            persistent=False,
        )
        self.register_buffer(
            "binaries",
            torch.ones(resolution, resolution, resolution, dtype=torch.bool),
            persistent=False,
        )

    @torch.no_grad()
    def update_occupancy(self, field: nn.Module, step: int) -> None:
        """Update cell densities from one jittered point per selected cell.

        Queries run in chunks of ``2**16``. Selected cells are updated with
        ``max(ema_decay * density, sigma)``, then the full grid is thresholded.
        Runs in training or eval mode; see :meth:`VolumeRenderer.update_occupancy`
        for intervals and warmup behavior.
        """
        if step % self.update_interval != 0:
            return
        resolution = self.grid_resolution
        device = self.density_grid.device
        num_cells = resolution**3
        if step < self.warmup_steps:
            cells = torch.arange(num_cells, device=device)
        else:
            # Discover new occupied regions while refreshing known ones.
            num_uniform = num_cells // 4
            uniform_cells = torch.randint(num_cells, (num_uniform,), device=device)
            occupied_cells = (
                self.binaries.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
            )
            if occupied_cells.numel() > num_uniform:
                indices = torch.randint(
                    occupied_cells.numel(), (num_uniform,), device=device
                )
                occupied_cells = occupied_cells[indices]
            cells = torch.cat([uniform_cells, occupied_cells])

        cell_indices = torch.stack(
            torch.unravel_index(cells, (resolution,) * 3), dim=-1
        )
        positions = (
            cell_indices.float() + torch.rand_like(cell_indices, dtype=torch.float32)
        ) / resolution
        box_min, box_max = self.aabb[:3], self.aabb[3:]
        positions = box_min + positions * (box_max - box_min)

        sampled_densities = torch.cat(
            [field.query_density(chunk) for chunk in positions.split(2**16)]
        )

        densities = self.density_grid.reshape(-1)
        densities[cells] = torch.maximum(
            densities[cells] * self.ema_decay, sampled_densities
        )
        threshold = min(self.density_threshold, densities.mean().item())
        self.binaries.copy_(self.density_grid > threshold)

    def _ray_aabb_intersect(
        self, rays_o: Tensor, rays_d: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return ``(R,)`` box-entry and exit distances, clipped to near/far.

        Uses the slab method. Rays that miss the box have an empty interval.
        """
        box_min, box_max = self.aabb[:3], self.aabb[3:]
        eps = 1e-10
        safe_directions = torch.where(
            rays_d.abs() > eps, rays_d, torch.full_like(rays_d, eps)
        )
        t1 = (box_min - rays_o) / safe_directions
        t2 = (box_max - rays_o) / safe_directions
        entry_distances = torch.minimum(t1, t2).amax(-1).clamp(min=self.near)
        exit_distances = torch.maximum(t1, t2).amin(-1).clamp(max=self.far)
        return entry_distances, torch.maximum(exit_distances, entry_distances)

    def forward(
        self,
        field: nn.Module,
        rays_o: Tensor,
        rays_d: Tensor,
        background: Tensor | None = None,
    ) -> dict[str, Tensor | int]:
        """Render a batch of rays through a radiance field.

        Allocates ``(R, num_marching_steps)`` marching buffers; in training
        mode each sample is jittered uniformly within its step, in eval mode
        it sits at the step midpoint. See :meth:`VolumeRenderer.forward` for
        the arguments and the returned dict.
        """
        num_rays = rays_o.shape[0]
        device = rays_o.device
        step_size = self.render_step_size
        if background is None:
            background = torch.ones(3, device=device)
        background = background.to(device)

        # Starting at each box entry keeps the grid within the diagonal budget.
        entry_distances, exit_distances = self._ray_aabb_intersect(rays_o, rays_d)
        num_marching_steps = self.num_marching_steps
        steps = torch.arange(num_marching_steps, device=device, dtype=torch.float32)
        offsets = (
            torch.rand(num_rays, num_marching_steps, device=device)
            if self.training
            else torch.full((1, num_marching_steps), 0.5, device=device)
        )
        distances = entry_distances[:, None] + (steps + offsets) * step_size
        sample_mask = distances < exit_distances[:, None]

        positions = rays_o[:, None, :] + distances[..., None] * rays_d[:, None, :]
        box_min, box_max = self.aabb[:3], self.aabb[3:]
        # Fold one axis at a time to avoid an (R, S, 3) index buffer.
        resolution = self.grid_resolution
        flat_indices = None
        for axis in range(3):
            indices = (
                (
                    (positions[..., axis] - box_min[axis])
                    / (box_max[axis] - box_min[axis])
                    * resolution
                )
                .int()
                .clamp_(0, resolution - 1)
            )
            # Flat indices can exceed int32 for large grids.
            flat_indices = (
                indices.long()
                if flat_indices is None
                else flat_indices * resolution + indices
            )
        sample_mask = sample_mask & self.binaries.reshape(-1)[flat_indices]
        del flat_indices
        if self.max_samples_per_ray is not None:
            # Keep the first occupied samples in front-to-back order.
            sample_mask = sample_mask & (
                sample_mask.cumsum(-1, dtype=torch.int32) <= self.max_samples_per_ray
            )

        sample_densities = torch.zeros(num_rays, num_marching_steps, device=device)
        sample_colors = torch.zeros(num_rays, num_marching_steps, 3, device=device)
        num_samples = int(sample_mask.sum().item())
        if num_samples > 0:
            points = positions[sample_mask]
            directions = rays_d[:, None, :].expand_as(positions)[sample_mask]
            # Free the (R, S, 3) buffer before querying the field.
            del positions
            colors, densities = field(points, directions)
            sample_densities[sample_mask] = densities
            sample_colors[sample_mask] = colors

        alphas = 1.0 - torch.exp(-sample_densities * step_size)
        transmittance = torch.cumprod(
            torch.cat(
                [torch.ones(num_rays, 1, device=device), 1.0 - alphas + 1e-10], dim=-1
            ),
            dim=-1,
        )[:, :-1]
        weights = alphas * transmittance

        opacity = weights.sum(-1, keepdim=True)
        rgb = (weights[..., None] * sample_colors).sum(-2) + (
            1.0 - opacity
        ) * background
        # Condition depth on a hit, matching nerfacc's rendering convention.
        depth = (weights * distances).sum(-1, keepdim=True) / opacity.clamp_min(
            torch.finfo(rgb.dtype).eps
        )
        return {
            "rgb": rgb,
            "opacity": opacity,
            "depth": depth,
            "num_samples": num_samples,
        }


class NerfAccRenderer(VolumeRenderer):
    """Volume renderer using ``nerfacc.OccGridEstimator`` and CUDA kernels.

    Requires a working nerfacc extension; see :func:`is_nerfacc_available`.
    Constructor arguments are documented in :class:`VolumeRenderer`.

    Unlike :class:`PyTorchRenderer`, the estimator starts empty and renders
    only background until :meth:`update_occupancy` runs. Sampling also queries
    density to stop rays at transmittance below ``1e-4`` before applying the
    sample cap. Grid buffers persist in checkpoints under ``estimator.``.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if nerfacc is None:
            raise ImportError(
                "nerfacc is not installed; use backend='pytorch' or "
                "pip install nerfacc"
            )
        self.estimator = nerfacc.OccGridEstimator(
            roi_aabb=self.aabb.tolist(),
            resolution=self.grid_resolution,
            levels=1,
        )

    def update_occupancy(self, field: nn.Module, step: int) -> None:
        """Update the grid through ``OccGridEstimator.update_every_n_steps``.

        Queries all selected cells in one batch. nerfacc requires training mode;
        see :meth:`VolumeRenderer.update_occupancy` for intervals and warmup.
        """
        self.estimator.update_every_n_steps(
            step=step,
            occ_eval_fn=lambda x: field.query_density(x).unsqueeze(-1),
            occ_thre=self.density_threshold,
            ema_decay=self.ema_decay,
            warmup_steps=self.warmup_steps,
            n=self.update_interval,
        )

    def forward(
        self,
        field: nn.Module,
        rays_o: Tensor,
        rays_d: Tensor,
        background: Tensor | None = None,
    ) -> dict[str, Tensor | int]:
        """Render a batch of rays through a radiance field.

        Samples are stratified in training mode. See :meth:`VolumeRenderer.forward`
        for arguments and outputs.
        """
        num_rays = rays_o.shape[0]
        if background is None:
            background = torch.ones(3, device=rays_o.device)
        background = background.to(rays_o.device)

        def midpoints(t_starts: Tensor, t_ends: Tensor, ray_indices: Tensor) -> Tensor:
            t_mid = (t_starts + t_ends)[:, None] / 2.0
            return rays_o[ray_indices] + rays_d[ray_indices] * t_mid

        def sigma_fn(t_starts: Tensor, t_ends: Tensor, ray_indices: Tensor) -> Tensor:
            return field.query_density(midpoints(t_starts, t_ends, ray_indices))

        def rgb_sigma_fn(
            t_starts: Tensor, t_ends: Tensor, ray_indices: Tensor
        ) -> tuple[Tensor, Tensor]:
            positions = midpoints(t_starts, t_ends, ray_indices)
            return field(positions, rays_d[ray_indices])

        ray_indices, t_starts, t_ends = self.estimator.sampling(
            rays_o,
            rays_d,
            sigma_fn=sigma_fn,
            near_plane=self.near,
            # nerfacc uses a finite far bound; box traversal still stops rays.
            far_plane=min(self.far, 1e10),
            render_step_size=self.render_step_size,
            stratified=self.training,
        )
        if self.max_samples_per_ray is not None and ray_indices.numel() > 0:
            # Samples are grouped by ray and ordered front to back.
            _, counts = torch.unique_consecutive(ray_indices, return_counts=True)
            segment_starts = torch.cumsum(counts, 0) - counts
            position_in_ray = torch.arange(
                ray_indices.shape[0], device=ray_indices.device
            ) - segment_starts.repeat_interleave(counts)
            keep = position_in_ray < self.max_samples_per_ray
            ray_indices = ray_indices[keep]
            t_starts = t_starts[keep]
            t_ends = t_ends[keep]
        rgb, opacity, depth, _ = nerfacc.rendering(
            t_starts,
            t_ends,
            ray_indices,
            n_rays=num_rays,
            rgb_sigma_fn=rgb_sigma_fn,
            render_bkgd=background,
        )
        return {
            "rgb": rgb,
            "opacity": opacity,
            "depth": depth,
            "num_samples": int(t_starts.shape[0]),
        }
