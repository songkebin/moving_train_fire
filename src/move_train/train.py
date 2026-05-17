from __future__ import annotations

import argparse
import csv
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from move_train.config import load_config, merge_overrides
from move_train.data import TrainFireDataset, compute_target_standardization, split_dataset
from move_train.model import MultiModalTransformer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def forward_batch(model: nn.Module, batch: dict[str, Any]) -> torch.Tensor:
    return model(
        image=batch["image"],
        env=batch["env"],
        speed=batch["speed"],
        hrr=batch["hrr"],
        position=batch["position"],
    )


def update_running_metrics(
    totals: dict[str, float],
    prediction: torch.Tensor,
    target: torch.Tensor,
    loss: torch.Tensor,
) -> None:
    batch_size = int(target.shape[0])
    elements = int(target.numel())
    error = prediction.detach() - target.detach()
    totals["samples"] += batch_size
    totals["elements"] += elements
    totals["loss"] += float(loss.detach().item()) * batch_size
    totals["absolute_error"] += float(error.abs().sum().item())
    totals["squared_error"] += float((error * error).sum().item())


def finalize_metrics(totals: dict[str, float]) -> dict[str, float]:
    samples = max(1.0, totals["samples"])
    elements = max(1.0, totals["elements"])
    mse = totals["squared_error"] / elements
    return {
        "loss": totals["loss"] / samples,
        "mae": totals["absolute_error"] / elements,
        "mse": mse,
        "rmse": math.sqrt(mse),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
    max_batches: int | None = None,
    target_mean: torch.Tensor | None = None,
    target_std: torch.Tensor | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {
        "samples": 0.0,
        "elements": 0.0,
        "loss": 0.0,
        "absolute_error": 0.0,
        "squared_error": 0.0,
    }

    description = "train" if training else "val"
    iterator = tqdm(loader, desc=description, leave=False)
    for batch_index, batch in enumerate(iterator, start=1):
        if max_batches is not None and batch_index > max_batches:
            break
        batch = batch_to_device(batch, device)

        if training:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                prediction = forward_batch(model, batch)
                loss = criterion(prediction, batch["target"])
            if scaler is not None and use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        else:
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=use_amp):
                    prediction = forward_batch(model, batch)
                    loss = criterion(prediction, batch["target"])

        metric_prediction = prediction
        metric_target = batch.get("target_raw", batch["target"])
        if target_mean is not None and target_std is not None:
            mean = target_mean.to(device=prediction.device, dtype=prediction.dtype)
            std = target_std.to(device=prediction.device, dtype=prediction.dtype)
            metric_prediction = prediction * std + mean
        update_running_metrics(totals, metric_prediction, metric_target, loss)
        metrics = finalize_metrics(totals)
        iterator.set_postfix(loss=f"{metrics['loss']:.4f}", rmse=f"{metrics['rmse']:.4f}")

    return finalize_metrics(totals)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    train_cfg: dict[str, Any],
    epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    scheduler_name = str(train_cfg.get("lr_scheduler", "cosine")).lower()
    if scheduler_name in {"", "none", "null", "false"}:
        return None
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=float(train_cfg.get("min_learning_rate", 1e-6)),
        )
    if scheduler_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(train_cfg.get("lr_scheduler_factor", 0.5)),
            patience=int(train_cfg.get("lr_scheduler_patience", 10)),
            min_lr=float(train_cfg.get("min_learning_rate", 1e-6)),
        )
    if scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(train_cfg.get("lr_step_size", 30)),
            gamma=float(train_cfg.get("lr_scheduler_factor", 0.1)),
        )
    raise ValueError(f"Unsupported lr_scheduler: {scheduler_name}")


def current_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def step_scheduler(
    scheduler: torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    val_metrics: dict[str, float],
) -> None:
    if scheduler is None:
        return
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(val_metrics["rmse"])
    else:
        scheduler.step()


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict[str, Any],
    metrics: dict[str, float],
    scheduler: torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "metrics": metrics,
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(payload, path)


def write_metrics_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_mae",
                "train_rmse",
                "val_loss",
                "val_mae",
                "val_rmse",
                "learning_rate",
            ],
        )
        writer.writeheader()


def append_metrics(
    path: Path,
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    learning_rate: float,
) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_mae",
                "train_rmse",
                "val_loss",
                "val_mae",
                "val_rmse",
                "learning_rate",
            ],
        )
        writer.writerow(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_mae": train_metrics["mae"],
                "train_rmse": train_metrics["rmse"],
                "val_loss": val_metrics["loss"],
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
                "learning_rate": learning_rate,
            }
        )


def train(config: dict[str, Any], device_name: str | None = None) -> dict[str, Any]:
    data_cfg = config["data"]
    train_cfg = config["train"]
    set_seed(int(data_cfg.get("random_seed", 42)))

    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = TrainFireDataset.from_config(config)
    train_dataset, val_dataset = split_dataset(
        dataset,
        val_fraction=float(data_cfg.get("val_fraction", 0.2)),
        seed=int(data_cfg.get("random_seed", 42)),
    )
    target_mean: torch.Tensor | None = None
    target_std: torch.Tensor | None = None
    target_cfg = data_cfg.get("target_standardization", {})
    if target_cfg.get("enabled", False):
        mean, std = compute_target_standardization(
            dataset.records,
            train_dataset.indices,
            epsilon=float(target_cfg.get("epsilon", 1e-6)),
        )
        dataset.set_target_standardization(mean, std)
        target_cfg["mode"] = target_cfg.get("mode", "per_sensor")
        target_cfg["mean"] = mean.tolist()
        target_cfg["std"] = std.tolist()
        target_cfg["fitted_on"] = "train_split"
        data_cfg["target_standardization"] = target_cfg
        target_mean = torch.from_numpy(mean).to(device=device).view(1, 5, 9)
        target_std = torch.from_numpy(std).to(device=device).view(1, 5, 9)

    pin_memory = bool(data_cfg.get("pin_memory", True)) and device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg.get("batch_size", 8)),
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(train_cfg.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=pin_memory,
    )

    model = MultiModalTransformer(**config["model"]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    epochs = int(train_cfg.get("epochs", 20))
    scheduler = build_scheduler(optimizer, train_cfg, epochs)
    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    output_dir = Path(train_cfg.get("output_dir", "outputs"))
    checkpoint_dir = Path(train_cfg.get("checkpoint_dir", output_dir / "checkpoints"))
    run_dir = output_dir / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    metrics_path = run_dir / "metrics.csv"
    write_metrics_header(metrics_path)

    best_rmse = float("inf")
    best_metrics: dict[str, float] | None = None
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        learning_rate = current_learning_rate(optimizer)
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
            max_batches=train_cfg.get("max_train_batches"),
            target_mean=target_mean,
            target_std=target_std,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
            scaler=None,
            use_amp=use_amp,
            max_batches=train_cfg.get("max_val_batches"),
            target_mean=target_mean,
            target_std=target_std,
        )
        append_metrics(metrics_path, epoch, train_metrics, val_metrics, learning_rate)
        step_scheduler(scheduler, val_metrics)
        save_checkpoint(
            checkpoint_dir / "last.pt", model, optimizer, epoch, config, val_metrics, scheduler
        )
        if val_metrics["rmse"] < best_rmse:
            best_rmse = val_metrics["rmse"]
            best_metrics = val_metrics
            save_checkpoint(
                checkpoint_dir / "best.pt", model, optimizer, epoch, config, val_metrics, scheduler
            )
        print(
            "lr={:.6g} train_loss={:.4f} train_rmse={:.4f} val_loss={:.4f} val_rmse={:.4f}".format(
                learning_rate,
                train_metrics["loss"],
                train_metrics["rmse"],
                val_metrics["loss"],
                val_metrics["rmse"],
            )
        )

    return {
        "metrics_path": metrics_path,
        "checkpoint_dir": checkpoint_dir,
        "best_metrics": best_metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the multimodal Transformer prototype.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--device", default=None, help="Torch device override, e.g. cuda or cpu.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Limit train batches for smoke tests.")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Limit val batches for smoke tests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    overrides: dict[str, Any] = {}
    if args.epochs is not None:
        overrides["train.epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["train.batch_size"] = args.batch_size
    if args.max_train_batches is not None:
        overrides["train.max_train_batches"] = args.max_train_batches
    if args.max_val_batches is not None:
        overrides["train.max_val_batches"] = args.max_val_batches
    config = merge_overrides(config, overrides)
    result = train(config, device_name=args.device)
    print(f"Metrics: {result['metrics_path']}")
    print(f"Checkpoints: {result['checkpoint_dir']}")


if __name__ == "__main__":
    main()
