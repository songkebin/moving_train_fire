from __future__ import annotations

import argparse
import csv
import math
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from move_train.config import load_config, merge_overrides
from move_train.data import TrainFireDataset, compute_target_standardization, split_dataset
from move_train.model import MultiModalTransformer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_device() -> torch.device:
    cuda_hidden = os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    if not cuda_hidden and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def forward_batch(model: nn.Module, batch: dict[str, Any]) -> torch.Tensor:
    kwargs = {
        "env": batch["env"],
        "speed": batch["speed"],
        "hrr": batch["hrr"],
        "position": batch["position"],
    }
    if "speed_value" in batch:
        kwargs["speed_value"] = batch["speed_value"]
    if "hrr_value" in batch:
        kwargs["hrr_value"] = batch["hrr_value"]
    if "image_features" in batch:
        kwargs["image_features"] = batch["image_features"]
        return model.forward_from_image_features(**kwargs)
    kwargs["image"] = batch["image"]
    return model(**kwargs)


class CachedImageFeatureDataset(Dataset):
    def __init__(self, dataset: TrainFireDataset, image_features: torch.Tensor) -> None:
        self.dataset = dataset
        self.image_features = image_features
        if len(dataset) != int(image_features.shape[0]):
            raise ValueError(
                f"Feature count {image_features.shape[0]} does not match dataset length {len(dataset)}."
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset.sample_without_image(index)
        sample["image_features"] = self.image_features[index].float()
        return sample


def cache_image_features(
    model: MultiModalTransformer,
    dataset: TrainFireDataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    description: str,
) -> torch.Tensor:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    disable_progress = os.environ.get("MOVE_TRAIN_DISABLE_TQDM", "").lower() in {
        "1",
        "true",
        "yes",
    }
    was_training = model.training
    model.eval()
    features: list[torch.Tensor] = []
    with torch.no_grad():
        iterator = tqdm(loader, desc=description, leave=False, disable=disable_progress)
        for batch in iterator:
            image = batch["image"].to(device, non_blocking=True)
            features.append(model.image_backbone(image).cpu())
    model.train(was_training)
    return torch.cat(features, dim=0)


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
    totals["target_sum"] += float(target.detach().sum().item())
    totals["target_squared_sum"] += float((target.detach() * target.detach()).sum().item())


def finalize_metrics(totals: dict[str, float]) -> dict[str, float]:
    samples = max(1.0, totals["samples"])
    elements = max(1.0, totals["elements"])
    mse = totals["squared_error"] / elements
    total_sum_of_squares = totals["target_squared_sum"] - (totals["target_sum"] ** 2 / elements)
    r2 = float("nan")
    if total_sum_of_squares > 0.0:
        r2 = 1.0 - totals["squared_error"] / total_sum_of_squares
    return {
        "loss": totals["loss"] / samples,
        "mae": totals["absolute_error"] / elements,
        "mse": mse,
        "rmse": math.sqrt(mse),
        "r2": r2,
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
    description: str | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {
        "samples": 0.0,
        "elements": 0.0,
        "loss": 0.0,
        "absolute_error": 0.0,
        "squared_error": 0.0,
        "target_sum": 0.0,
        "target_squared_sum": 0.0,
    }

    progress_description = description or ("train" if training else "val")
    disable_progress = os.environ.get("MOVE_TRAIN_DISABLE_TQDM", "").lower() in {
        "1",
        "true",
        "yes",
    }
    iterator = tqdm(loader, desc=progress_description, leave=False, disable=disable_progress)
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
                "train_r2",
                "val_loss",
                "val_mae",
                "val_rmse",
                "val_r2",
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
                "train_r2",
                "val_loss",
                "val_mae",
                "val_rmse",
                "val_r2",
                "learning_rate",
            ],
        )
        writer.writerow(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_mae": train_metrics["mae"],
                "train_rmse": train_metrics["rmse"],
                "train_r2": train_metrics["r2"],
                "val_loss": val_metrics["loss"],
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
                "val_r2": val_metrics["r2"],
                "learning_rate": learning_rate,
            }
        )


def write_split_metrics(path: Path, split: str, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "loss", "mae", "mse", "rmse", "r2"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "split": split,
                "loss": metrics["loss"],
                "mae": metrics["mae"],
                "mse": metrics["mse"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
            }
        )


def train(config: dict[str, Any], device_name: str | None = None) -> dict[str, Any]:
    data_cfg = config["data"]
    train_cfg = config["train"]
    set_seed(int(data_cfg.get("random_seed", 42)))

    device = torch.device(device_name) if device_name is not None else default_device()
    split_cfg = data_cfg.get("splits", {})
    uses_explicit_splits = bool(data_cfg.get("manifest")) and "train" in split_cfg and "val" in split_cfg
    if uses_explicit_splits:
        train_dataset = TrainFireDataset.from_config(config, split_path=split_cfg["train"])
        val_dataset = TrainFireDataset.from_config(config, split_path=split_cfg["val"])
        target_records = train_dataset.records
        target_indices = range(len(train_dataset.records))
        datasets_for_standardization = [train_dataset, val_dataset]
    else:
        dataset = TrainFireDataset.from_config(config)
        train_dataset, val_dataset = split_dataset(
            dataset,
            val_fraction=float(data_cfg.get("val_fraction", 0.2)),
            seed=int(data_cfg.get("random_seed", 42)),
        )
        target_records = dataset.records
        target_indices = train_dataset.indices
        datasets_for_standardization = [dataset]

    target_mean: torch.Tensor | None = None
    target_std: torch.Tensor | None = None
    target_cfg = data_cfg.get("target_standardization", {})
    if (
        target_cfg.get("enabled", False)
        and target_cfg.get("from_init_checkpoint", False)
        and ("mean" not in target_cfg or "std" not in target_cfg)
    ):
        init_checkpoint = train_cfg.get("init_checkpoint")
        if not init_checkpoint:
            raise ValueError(
                "target_standardization.from_init_checkpoint requires train.init_checkpoint."
            )
        checkpoint = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
        checkpoint_target_cfg = (
            checkpoint.get("config", {}).get("data", {}).get("target_standardization", {})
        )
        if "mean" not in checkpoint_target_cfg or "std" not in checkpoint_target_cfg:
            raise ValueError(
                f"Initial checkpoint does not contain target standardization stats: {init_checkpoint}"
            )
        target_cfg["mean"] = checkpoint_target_cfg["mean"]
        target_cfg["std"] = checkpoint_target_cfg["std"]
        target_cfg["fitted_on"] = f"init_checkpoint:{init_checkpoint}"
    if target_cfg.get("enabled", False):
        if "mean" in target_cfg and "std" in target_cfg:
            mean = np.asarray(target_cfg["mean"], dtype=np.float32).reshape(5, 9)
            std = np.asarray(target_cfg["std"], dtype=np.float32).reshape(5, 9)
            fitted_on = str(target_cfg.get("fitted_on", "configured"))
        else:
            mean, std = compute_target_standardization(
                target_records,
                target_indices,
                epsilon=float(target_cfg.get("epsilon", 1e-6)),
            )
            fitted_on = "train_split"
        for target_dataset in datasets_for_standardization:
            target_dataset.set_target_standardization(mean, std)
        target_cfg["mode"] = target_cfg.get("mode", "per_sensor")
        target_cfg["mean"] = mean.tolist()
        target_cfg["std"] = std.tolist()
        target_cfg["fitted_on"] = fitted_on
        data_cfg["target_standardization"] = target_cfg
        target_mean = torch.from_numpy(mean).to(device=device).view(1, 5, 9)
        target_std = torch.from_numpy(std).to(device=device).view(1, 5, 9)

    test_dataset = None
    if uses_explicit_splits and "test" in split_cfg:
        test_dataset = TrainFireDataset.from_config(config, split_path=split_cfg["test"])

    pin_memory = bool(data_cfg.get("pin_memory", True)) and device.type == "cuda"
    model = MultiModalTransformer(**config["model"]).to(device)
    init_checkpoint = train_cfg.get("init_checkpoint")
    if init_checkpoint:
        checkpoint = torch.load(init_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded initial checkpoint: {init_checkpoint}")

    cache_frozen_image_features = (
        bool(train_cfg.get("cache_frozen_image_features", False))
        and isinstance(train_dataset, TrainFireDataset)
        and isinstance(val_dataset, TrainFireDataset)
        and not model.backbone_trainable
    )
    if cache_frozen_image_features:
        batch_size = int(train_cfg.get("batch_size", 8))
        num_workers = int(data_cfg.get("num_workers", 4))
        print("Caching frozen image features for train split.")
        train_dataset = CachedImageFeatureDataset(
            train_dataset,
            cache_image_features(
                model,
                train_dataset,
                device,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                description="cache-train",
            ),
        )
        print("Caching frozen image features for val split.")
        val_dataset = CachedImageFeatureDataset(
            val_dataset,
            cache_image_features(
                model,
                val_dataset,
                device,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                description="cache-val",
            ),
        )

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
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=int(train_cfg.get("batch_size", 8)),
            shuffle=False,
            num_workers=int(data_cfg.get("num_workers", 4)),
            pin_memory=pin_memory,
        )

    criterion = nn.MSELoss()
    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    if not trainable_parameters:
        raise ValueError("Model has no trainable parameters.")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
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
    best_checkpoint_path = checkpoint_dir / "best.pt"
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
                best_checkpoint_path, model, optimizer, epoch, config, val_metrics, scheduler
            )
        print(
            "lr={:.6g} train_loss={:.4f} train_rmse={:.4f} train_r2={:.4f} "
            "val_loss={:.4f} val_rmse={:.4f} val_r2={:.4f}".format(
                learning_rate,
                train_metrics["loss"],
                train_metrics["rmse"],
                train_metrics["r2"],
                val_metrics["loss"],
                val_metrics["rmse"],
                val_metrics["r2"],
            )
        )

    test_metrics: dict[str, float] | None = None
    test_metrics_path: Path | None = None
    if test_loader is not None:
        checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_metrics = run_epoch(
            model,
            test_loader,
            criterion,
            device,
            optimizer=None,
            scaler=None,
            use_amp=use_amp,
            max_batches=train_cfg.get("max_test_batches"),
            target_mean=target_mean,
            target_std=target_std,
            description="test",
        )
        test_metrics_path = run_dir / "test_metrics.csv"
        write_split_metrics(test_metrics_path, "test", test_metrics)
        print(
            "test_loss={:.4f} test_rmse={:.4f} test_r2={:.4f}".format(
                test_metrics["loss"],
                test_metrics["rmse"],
                test_metrics["r2"],
            )
        )

    return {
        "metrics_path": metrics_path,
        "test_metrics_path": test_metrics_path,
        "checkpoint_dir": checkpoint_dir,
        "best_metrics": best_metrics,
        "test_metrics": test_metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the multimodal Transformer prototype.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--device", default=None, help="Torch device override, e.g. cuda or cpu.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Limit train batches for smoke tests.")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Limit val batches for smoke tests.")
    parser.add_argument("--max-test-batches", type=int, default=None, help="Limit test batches for smoke tests.")
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
    if args.max_test_batches is not None:
        overrides["train.max_test_batches"] = args.max_test_batches
    config = merge_overrides(config, overrides)
    result = train(config, device_name=args.device)
    print(f"Metrics: {result['metrics_path']}")
    if result["test_metrics_path"] is not None:
        print(f"Test metrics: {result['test_metrics_path']}")
    print(f"Checkpoints: {result['checkpoint_dir']}")


if __name__ == "__main__":
    main()
