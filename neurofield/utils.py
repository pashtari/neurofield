"""Module construction, experiment logging, serialization and image operators."""

import gzip
import io
import json
import logging
import os
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import datetime, timezone
from numbers import Number
from typing import Any, Literal, TypeAlias

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = [
    "ModuleSpec",
    "build_module",
    "count_parameters",
    "JSONLogger",
    "setup_experiment_loggers",
    "iter_json_records",
    "read_json_records",
    "encode_dict",
    "decode_dict",
    "read_config",
    "save_results",
    "add_sensor_noise",
    "sobel_gradients",
    "radon",
    "render_mesh",
]


# Constructor forms accepted by build_module; JSON lists also work at runtime.
ModuleSpec: TypeAlias = (
    nn.Module | str | type[nn.Module] | tuple[str | type[nn.Module], dict[str, Any]]
)


def build_module(
    spec: ModuleSpec,
    registry: Mapping[str, type[nn.Module]],
    *args: Any,
    **kwargs: Any,
) -> nn.Module:
    """Build a module from an instance, registry key, class, or ``(key/class, params)``.

    A two-element list also works, so specs can be loaded from JSON. Instances
    are returned unchanged. Otherwise, the resolved class receives ``*args``,
    ``**params`` and ``**kwargs``; duplicate keyword arguments raise TypeError.

    Args:
        spec: Module instance, key, class, or a key/class paired with parameters.
        registry: Mapping from string keys to module classes.
        *args: Positional arguments passed to the constructor.
        **kwargs: Additional constructor arguments.

    Example::

        build_module(("linear", {"bias": False}), {"linear": nn.Linear}, 4, 2)
    """
    if isinstance(spec, nn.Module):
        return spec

    if isinstance(spec, (str, type)):
        key_or_class, params = spec, {}
    elif isinstance(spec, (tuple, list)) and len(spec) == 2:
        key_or_class, params = spec
    else:
        raise ValueError(
            "spec must be an nn.Module instance, a registry key / nn.Module "
            f"subclass, or a (name/type, params) pair, got {spec!r}"
        )

    if isinstance(key_or_class, str):
        if key_or_class not in registry:
            known = ", ".join(sorted(registry)) or "none"
            raise KeyError(
                f"unknown key {key_or_class!r}; known keys: {known}. Pass an "
                "nn.Module subclass or instance instead."
            )
        module_class = registry[key_or_class]
    else:
        module_class = key_or_class
    return module_class(*args, **params, **kwargs)


def _average_dicts(dicts: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Group values by key, averaging numeric groups and keeping others as lists.

    Returns a defaultdict with a list factory; input keys may differ.
    """
    groups: defaultdict[str, Any] = defaultdict(list)

    for record in dicts:
        for key, value in record.items():
            groups[key].append(value)

    for key, values in groups.items():
        if all(isinstance(value, Number) for value in values):
            groups[key] = sum(values) / len(values)

    return groups


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count parameters as real numbers, counting complex elements twice.

    Only parameters with ``requires_grad=True`` count when ``trainable_only``
    is set. Counting real components makes real and complex models comparable.
    """
    return sum(
        param.numel() * (2 if param.is_complex() else 1)
        for param in model.parameters()
        if param.requires_grad or not trainable_only
    )


class JSONLogger:
    """Append timestamped records to a JSON file.

    Records are pretty-printed JSON objects separated by newlines. Read them
    with :func:`read_json_records` or :func:`iter_json_records`. A missing or
    empty ``path`` disables writing.
    """

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = path

    def log(self, entry: Mapping[str, Any]) -> None:
        """Append a JSON-serializable record with a UTC ISO 8601 timestamp.

        An explicit ``timestamp`` in ``entry`` takes precedence.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        entry = {"timestamp": timestamp, **entry}
        if self.path:
            with open(self.path, "a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(entry, ensure_ascii=False, indent=4) + "\n")


def setup_experiment_loggers(
    log_dir: str | os.PathLike[str] | None,
    logger_name: str,
    log_filename: str = "log.txt",
    json_filename: str = "log.json",
    mode: Literal["w", "a"] = "w",
) -> tuple[logging.Logger, JSONLogger]:
    """Configure console, text and JSON logging for an experiment.

    Replaces the named logger's handlers and disables propagation. Text files
    append after an optional initial truncation, allowing training and
    evaluation loggers to share a file. JSON records always append.

    Args:
        log_dir: Output directory, created if needed; ``None`` logs to console only.
        logger_name: Name passed to :func:`logging.getLogger`.
        log_filename: Text log filename within ``log_dir``.
        json_filename: JSON log filename within ``log_dir``.
        mode: ``"w"`` truncates the text log once; ``"a"`` appends.

    Returns:
        The text logger and JSON logger (disabled when ``log_dir`` is absent).
    """
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_filename)
        json_log_path = os.path.join(log_dir, json_filename)
    else:
        log_path = None
        json_log_path = None

    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_path:
        if mode == "w":
            # Append thereafter so training and evaluation can share the file.
            open(log_path, "w", encoding="utf-8").close()
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    json_logger = JSONLogger(json_log_path)

    if log_dir:
        logger.info("Logging to %s", log_dir)

    return logger, json_logger


def iter_json_records(path: str | os.PathLike[str]) -> Iterator[dict[str, Any]]:
    """Yield records from a stream of JSON objects, including pretty-printed logs.

    Parsing stops at the first malformed or incomplete record, so an
    interrupted final write is ignored.
    """
    decoder = json.JSONDecoder()
    with open(path, "r", encoding="utf-8") as log_file:
        buffer = ""
        for line in log_file:
            buffer += line
            while buffer:
                buffer = buffer.lstrip()
                try:
                    record, end = decoder.raw_decode(buffer)
                    yield record
                    buffer = buffer[end:]
                except json.JSONDecodeError:
                    break


def read_json_records(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Read all records yielded by :func:`iter_json_records`."""
    return list(iter_json_records(path))


def sobel_gradients(
    image: Tensor,
    normalize: bool = True,
    padding_mode: Literal["constant", "reflect", "replicate", "circular"] = "replicate",
) -> tuple[Tensor, Tensor]:
    """Return per-channel Sobel gradients ``(gx, gy)`` of a channels-last batch.

    Input and outputs have shape ``(B, H, W, C)``; computation uses float32.
    ``gx`` points along increasing columns and ``gy`` along increasing rows.

    Args:
        image: Image batch in channels-last layout.
        normalize: Divide kernels by 8 so a unit ramp has unit gradient.
        padding_mode: Mode for one-pixel padding; ``"constant"`` pads with zeros.
    """
    x = image.movedim(-1, 1).float()
    channels = x.shape[1]

    gx_kernel = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=x.dtype,
        device=x.device,
    )
    if normalize:
        gx_kernel = gx_kernel / 8.0

    padded = F.pad(x, (1, 1, 1, 1), mode=padding_mode)
    gradients = [
        F.conv2d(padded, kernel.expand(channels, 1, 3, 3), groups=channels)
        for kernel in (gx_kernel, gx_kernel.t())
    ]
    return gradients[0].movedim(1, -1), gradients[1].movedim(1, -1)


def encode_dict(data: dict[str, Any], compress: bool = True) -> bytes:
    """Serialize a tensor dictionary to bytes, optionally with gzip compression.

    Uses :func:`torch.save`. To round-trip through :func:`decode_dict`, values
    must be tensors, primitive types or standard containers accepted by
    ``torch.load(weights_only=True)``.
    """
    buffer = io.BytesIO()
    torch.save(data, buffer)
    raw = buffer.getvalue()
    return gzip.compress(raw) if compress else raw


def decode_dict(encoded: bytes, compressed: bool = True) -> dict[str, Any]:
    """Deserialize :func:`encode_dict` output, loading tensors onto the CPU.

    ``compressed`` must match the encoder's ``compress`` argument. Loading uses
    PyTorch's default ``weights_only=True``.
    """
    raw = gzip.decompress(encoded) if compressed else encoded
    return torch.load(io.BytesIO(raw), map_location="cpu")


def read_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read a UTF-8 JSON configuration file."""
    with open(path, "r", encoding="utf-8") as json_file:
        return json.load(json_file)


def save_results(
    results: Sequence[dict[str, Any]],
    save_dir: str | os.PathLike[str],
    prefix: str = "",
) -> None:
    """Write records to ``save_dir/{prefix}_results.json``, replacing any existing file.

    Creates the directory if needed.
    """
    os.makedirs(save_dir, exist_ok=True)
    with open(
        os.path.join(save_dir, f"{prefix}_results.json"), "w", encoding="utf-8"
    ) as json_file:
        json.dump(results, json_file, indent=4)


def add_sensor_noise(
    image: Tensor,
    readout_noise_std: float = 1.0,
    integration_time: float = 1.0,
    seed: int = 0,
) -> Tensor:
    """Simulate photon shot noise and Gaussian readout noise.

    The image is converted to float32, scaled to photon counts, Poisson-sampled,
    then perturbed by Gaussian noise. The result is divided by integration time
    and clamped to ``[0, 1]``, with the same shape and device as the input.

    Args:
        image: Image with values in ``[0, 1]``.
        readout_noise_std: Gaussian standard deviation in count units.
        integration_time: Photon-count scale; longer exposures reduce relative noise.
        seed: Reseeds the global PyTorch RNG, affecting subsequent random draws.
    """
    torch.manual_seed(seed)
    # Sample in float32 even when the input uses a lower-precision dtype.
    shot_noise = torch.poisson(image.float() * integration_time)
    readout_noise = torch.randn_like(shot_noise) * readout_noise_std
    noisy = (shot_noise + readout_noise) / integration_time
    return noisy.clamp(0.0, 1.0)


def render_mesh(
    mesh_path: str | os.PathLike[str],
    image_path: str | os.PathLike[str],
    width: int = 1920,
    height: int = 1080,
) -> None:
    """Render a mesh to an image using a hidden Open3D window.

    Requires the ``3d`` extra. Trimesh loads ``mesh_path`` and joins scenes into
    one mesh; Open3D renders the default view at ``width`` by ``height`` pixels.
    ``image_path`` determines the output format.
    """
    import open3d as o3d
    import trimesh

    mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    o3d_mesh.compute_vertex_normals()

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(width=width, height=height, visible=False)
    visualizer.add_geometry(o3d_mesh)
    visualizer.capture_screen_image(str(image_path), do_render=True)
    visualizer.destroy_window()


def radon(image: Tensor, angles: Tensor) -> Tensor:
    """Compute a sinogram by rotating the image and summing along its height.

    Args:
        image: Tensor of shape ``(1, C, H, W)``.
        angles: Angles in degrees, shape ``(A,)``, on the image device.

    Returns:
        Shape ``(A, C, W)`` with singleton dimensions removed. Rotations use
        Kornia's bilinear interpolation and zero padding.
    """
    import kornia

    rotated = kornia.geometry.rotate(image.repeat_interleave(len(angles), 0), angles)
    return rotated.sum(2).squeeze()
