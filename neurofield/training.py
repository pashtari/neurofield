"""Training loops for neural fields on coordinate datasets."""

import copy
import os
import time
from collections.abc import Callable, Mapping
from typing import Any, TypeAlias

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .evaluation import evaluate
from .quantization import uniform_quantize
from .utils import _average_dicts, count_parameters, setup_experiment_loggers

__all__ = ["train", "train_epoch"]

LossFn: TypeAlias = Callable[
    [dict[str, Any], nn.Module], tuple[Tensor, dict[str, float]]
]


def _mse_loss(
    batch: dict[str, Any], model: nn.Module
) -> tuple[Tensor, dict[str, float]]:
    """Default reconstruction loss, with PSNR for targets in ``[-1, 1]``."""
    output = model(batch["input"])
    mse = F.mse_loss(output, batch["target"])
    psnr = 10 * torch.log10(4.0 / mse)
    return mse, {"mse": mse.detach().item(), "psnr": psnr.detach().item()}


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: LossFn,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Run one training epoch, returning the last batch's metrics and ID.

    Tensor entries without an underscore-prefixed key are moved to ``device``.
    ``loss_fn(batch, model)`` returns a scalar loss and float metrics. Optimizer
    and scheduler each step once per batch. The model is left in training mode.
    ``device=None`` selects CUDA when available, else CPU.
    """
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.train()

    last_metrics: dict[str, Any] = {}
    last_id: Any = None

    for batch in dataloader:
        last_id = batch.get("id")
        for key, value in batch.items():
            if isinstance(value, Tensor) and not key.startswith("_"):
                batch[key] = value.to(device)

        loss, metrics = loss_fn(batch, model)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        last_metrics = metrics

    return {"id": last_id, **last_metrics}


def train(
    model: nn.Module,
    train_dataset: Dataset,
    eval_dataset: Dataset | None = None,
    *,
    loss_fn: LossFn | None = None,
    batch_size: int = 1,
    num_epochs: int = 1000,
    lr: float = 0.01,
    weight_decay: float = 0.0,
    adam_betas: tuple[float, float] = (0.9, 0.999),
    adam_eps: float = 1e-8,
    quantize: bool = False,
    quant_max: int = 127,
    quant_interval: int = -1,
    metrics: Mapping[str, Callable[[Tensor, Tensor], float]] | None = None,
    rate_metrics: Mapping[str, Callable[[bytes, Tensor], float]] | None = None,
    log_interval: int = 1,
    eval_interval: int | None = None,
    chunk_size: int | None = None,
    save_reconstruction: bool = True,
    device: str | torch.device | None = None,
    ckpt_path: str | os.PathLike[str] | None = None,
    log_dir: str | os.PathLike[str] | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Fit a neural field with Adam and a cosine-annealed learning rate.

    Moves model and supported datasets to ``device`` in place. The scheduler
    steps per batch with ``T_max=num_epochs`` and a minimum rate of ``lr / 100``.
    Optional quantization modifies parameters during training, evaluates a
    quantized copy and quantizes the final model in place.

    Args:
        model: Neural field to train in place.
        train_dataset: Dataset returning ``input`` and ``target`` entries.
        eval_dataset: Evaluation dataset; ``None`` or empty uses ``train_dataset``.
        loss_fn: Called as ``loss_fn(batch, model) -> (loss, metrics)``.
            ``None`` uses MSE and logs PSNR assuming targets in ``[-1, 1]``.
        batch_size: Training batch size.
        num_epochs: Number of training epochs.
        lr: Initial learning rate.
        weight_decay: Adam weight decay.
        adam_betas: Adam moment decay rates; hash grids use ``(0.9, 0.99)``.
        adam_eps: Adam epsilon; hash grids use ``1e-15``.
        quantize: Enable quantized evaluation and final in-place quantization.
        quant_max: Largest integer quantization level.
        quant_interval: Epochs between intermediate fake-quantization steps.
            Non-positive values disable these steps; requires ``quantize=True``.
        metrics: Distortion metrics passed to :func:`evaluate`.
        rate_metrics: Rate metrics passed to :func:`evaluate`.
        log_interval: Epochs between progress, file and history logging.
            Non-positive values disable epoch logging.
        eval_interval: Epochs between evaluations. ``None`` or zero evaluates
            at the end; negative values disable evaluation.
        chunk_size: Coordinates per evaluation forward pass; ``None`` uses all.
        save_reconstruction: Save each evaluation's reconstruction under
            ``log_dir``; see :func:`evaluate`.
        device: Training device; ``None`` selects CUDA when available, else CPU.
        ckpt_path: State dict to load first, if the file exists.
        log_dir: Directory for logs, reconstructions and ``checkpoint.pt``.
            Text logs are truncated initially; JSON records append.
            ``None`` logs to console only.
        seed: Seed for the global PyTorch RNG.

    Returns:
        ``config`` with logged settings, ``model_dict`` with final parameters
        (integer codes and scales when quantized), and ``history`` with one
        record per logged epoch. Records contain last-batch training metrics
        and averaged evaluation metrics under ``eval`` when evaluated.
    """
    logger, json_logger = setup_experiment_loggers(log_dir, __name__, mode="w")
    config: dict[str, Any] = {}

    torch.manual_seed(seed)
    logger.info("Seed: %d", seed)
    config["seed"] = seed

    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    config["device"] = str(device)

    if not eval_dataset:
        eval_dataset = train_dataset

    # Keep coordinates on the training device so subsampling runs there too.
    if hasattr(train_dataset, "to"):
        train_dataset.to(device)
    if hasattr(eval_dataset, "to"):
        eval_dataset.to(device)

    on_cuda = getattr(getattr(train_dataset, "input", None), "is_cuda", False)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        pin_memory=not on_cuda,
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

    if loss_fn is None:
        loss_fn = _mse_loss

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        betas=adam_betas,
        eps=adam_eps,
        weight_decay=weight_decay,
    )
    logger.info("Optimizer: %s", type(optimizer).__name__)
    config["optimizer"] = type(optimizer).__name__
    logger.info("Learning rate: %f", lr)
    config["lr"] = lr
    logger.info("Weight decay: %f", weight_decay)
    config["weight_decay"] = weight_decay
    config["adam_betas"] = tuple(adam_betas)
    config["adam_eps"] = adam_eps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr / 100
    )
    logger.info("Scheduler: %s", type(scheduler).__name__)
    config["scheduler"] = type(scheduler).__name__

    chunk_size = chunk_size or -1
    logger.info("Chunk size: %d", chunk_size)
    config["chunk_size"] = chunk_size

    logger.info("Starting training for %d epochs", num_epochs)
    config["num_epochs"] = num_epochs

    eval_interval = eval_interval or num_epochs
    progress_bar = tqdm(range(1, num_epochs + 1), desc="Training")

    history: list[dict[str, Any]] = []

    elapsed_time = 0.0
    for epoch in progress_bar:
        epoch_start_time = time.time()

        epoch_results = train_epoch(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )

        if quantize and quant_interval > 0 and epoch % quant_interval == 0:
            model, _ = uniform_quantize(model, quant_max)

        if torch.device(device).type == "cuda":
            # CUDA kernels run asynchronously; wait so timings are wall-clock.
            torch.cuda.synchronize(device)
        epoch_duration = time.time() - epoch_start_time
        elapsed_time += epoch_duration

        results = {
            "epoch": epoch,
            "elapsed": elapsed_time,
            "duration": epoch_duration,
            **epoch_results,
        }

        if eval_interval > 0 and epoch % eval_interval == 0:
            if quantize:
                eval_model, model_dict = uniform_quantize(
                    copy.deepcopy(model), quant_max
                )
            else:
                eval_model = model
                model_dict = eval_model.state_dict()

            eval_results = evaluate(
                model=eval_model,
                dataset=eval_dataset,
                model_dict=model_dict,
                metrics=metrics,
                rate_metrics=rate_metrics,
                chunk_size=chunk_size,
                device=device,
                log_dir=log_dir,
                save_reconstruction=save_reconstruction,
            )
            results["eval"] = _average_dicts(eval_results["history"])

        if log_interval > 0 and epoch % log_interval == 0:
            bar_metrics = {
                name: f"{value:.4f}"
                for name, value in results.items()
                if isinstance(value, float)
            }
            progress_bar.set_postfix(bar_metrics)

            metrics_str = ", ".join(
                f"{key}={value}" for key, value in bar_metrics.items()
            )
            progress_percent = (epoch / num_epochs) * 100
            logger.info(
                "Training - Epoch [%d/%d], Progress %.1f%% - %s",
                epoch,
                num_epochs,
                progress_percent,
                metrics_str,
            )
            json_logger.log(results)
            history.append(results)

    logger.info("Training completed")

    if quantize:
        model, model_dict = uniform_quantize(model, quant_max)
    else:
        model_dict = model.state_dict()

    if log_dir:
        save_ckpt_path = os.path.join(log_dir, "checkpoint.pt")
        torch.save(model.state_dict(), save_ckpt_path)
        logger.info("Saved checkpoint to %s", save_ckpt_path)

    return {"config": config, "model_dict": model_dict, "history": history}
