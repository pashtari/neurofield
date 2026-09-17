"""Chunked inference and dataset evaluation for neural fields."""

import os
import time
from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from .utils import count_parameters, encode_dict, setup_experiment_loggers

__all__ = ["evaluate", "chunked_inference"]


@torch.no_grad()
def chunked_inference(
    model: nn.Module,
    x: Tensor,
    *,
    chunk_size: int | None = None,
    device: str | torch.device | None = None,
) -> Tensor:
    """Evaluate coordinates in chunks under ``torch.no_grad()``.

    Moves ``model`` to ``device`` and leaves it in evaluation mode. Leading
    input dimensions are flattened for chunking, then restored in the output.

    Args:
        model: Maps ``(N, in_features)`` to ``(N, out_features)``.
        x: Coordinates of shape ``(*, in_features)``.
        chunk_size: Rows per forward pass. ``None`` or a non-positive value
            evaluates the full input without flattening.
        device: Inference device; ``None`` selects CUDA when available, else CPU.

    Returns:
        Model output of shape ``(*, out_features)`` on ``device``.
    """
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    if chunk_size is None or chunk_size <= 0:
        return model(x.to(device))

    *batch_shape, _ = x.shape
    x = x.flatten(0, -2)

    outputs = [model(chunk.to(device)) for chunk in x.split(chunk_size, dim=0)]
    return torch.cat(outputs, dim=0).reshape(*batch_shape, -1)


def evaluate(
    model: nn.Module,
    dataset: Dataset,
    *,
    model_dict: dict[str, Any] | None = None,
    metrics: Mapping[str, Callable[[Tensor, Tensor], float]] | None = None,
    rate_metrics: Mapping[str, Callable[[bytes, Tensor], float]] | None = None,
    chunk_size: int | None = None,
    device: str | torch.device = "cpu",
    ckpt_path: str | os.PathLike[str] | None = None,
    log_dir: str | os.PathLike[str] | None = None,
    save_reconstruction: bool = True,
) -> dict[str, Any]:
    """Reconstruct each dataset item, compute metrics and optionally save outputs.

    Items must contain ``id``, ``input`` and ``_original``. Model output is
    mapped through ``dataset.postprocess`` when available, then moved to CPU
    for metrics. The model is moved to ``device`` and left in evaluation mode.

    Args:
        model: Trained neural field.
        dataset: Dataset to evaluate one item at a time.
        model_dict: Parameter dictionary encoded for rate metrics; ``None``
            uses ``model.state_dict()``.
        metrics: Functions called as ``fn(reconstruction, original)``.
        rate_metrics: Functions called as ``fn(encoded_bytes, original)``.
        chunk_size: Coordinates per forward pass; see :func:`chunked_inference`.
        device: Evaluation device.
        ckpt_path: State dict to load first, if the file exists.
        log_dir: Directory for appended logs and reconstructions, created if
            needed. ``None`` logs to console only.
        save_reconstruction: Save with ``dataset.save`` under
            ``log_dir/{id}_reconstructed`` when a log directory is given.

    Returns:
        ``config`` with model name, parameter count and chunk size; ``history``
        with one record per sample containing its ID, elapsed time, duration
        and metric values.
    """
    model_dict = model_dict or model.state_dict()
    metrics = metrics or {}
    rate_metrics = rate_metrics or {}

    logger, json_logger = setup_experiment_loggers(log_dir, __name__, mode="a")
    config: dict[str, Any] = {}

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        pin_memory=not getattr(getattr(dataset, "input", None), "is_cuda", False),
        num_workers=0,
    )

    if ckpt_path and os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)

    config["model"] = type(model).__name__
    logger.info("Model: %s", config["model"])

    num_params = count_parameters(model)
    logger.info("Learnable parameters: %d", num_params)
    config["num_params"] = num_params

    chunk_size = chunk_size or -1
    logger.info("Chunk size: %d", chunk_size)
    config["chunk_size"] = chunk_size

    # Parameters are fixed during evaluation, so every sample shares one bitstream.
    encoded = encode_dict(model_dict) if rate_metrics else b""
    postprocess = getattr(dataset, "postprocess", None)

    history: list[dict[str, Any]] = []
    start_time = time.time()
    for i, batch in enumerate(dataloader):
        item_id = batch["id"][0]
        x = batch["input"].to(device)
        original = batch["_original"][0]

        sample_start_time = time.time()

        output = chunked_inference(model, x, chunk_size=chunk_size, device=device)
        reconstruction = (
            torch.vmap(postprocess)(output) if postprocess is not None else output
        )
        reconstruction = reconstruction.cpu()[0]

        sample_duration = time.time() - sample_start_time
        elapsed_time = time.time() - start_time

        results = {"id": item_id, "elapsed": elapsed_time, "duration": sample_duration}

        for metric_name, metric_fn in metrics.items():
            metric_value = metric_fn(reconstruction, original)
            logger.info("%s: %.4f", metric_name, metric_value)
            results[metric_name] = metric_value

        for metric_name, metric_fn in rate_metrics.items():
            metric_value = metric_fn(encoded, original)
            logger.info("%s: %.4f", metric_name, metric_value)
            results[metric_name] = metric_value

        metrics_str = ", ".join(
            f"{key}={value:.4f}"
            for key, value in results.items()
            if isinstance(value, float)
        )
        logger.info(
            "Evaluation - Sample [%d/%d], ID %s - %s",
            i + 1,
            len(dataloader),
            item_id,
            metrics_str,
        )
        json_logger.log(results)

        history.append(results)

        if log_dir and save_reconstruction:
            reconstruction_path = os.path.join(log_dir, f"{item_id}_reconstructed")
            dataset.save(reconstruction, reconstruction_path)
            logger.info("Reconstructed signal saved to %s", reconstruction_path)

    return {"config": config, "history": history}
