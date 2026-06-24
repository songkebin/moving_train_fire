from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from move_train.data import TEMPERATURE_COLUMNS, TrainFireDataset, inverse_standardize_temperature
from move_train.model import MultiModalTransformer


PHYSICAL_COLUMNS = ["e", "v", "q", "s"]


def load_checkpoint_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[MultiModalTransformer, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = copy.deepcopy(checkpoint["config"])
    config["model"]["pretrained_backbone"] = False
    model = MultiModalTransformer(**config["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config


def forward_batch(
    model: MultiModalTransformer,
    batch: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    kwargs = {
        "env": batch["env"].to(device, non_blocking=True),
        "speed": batch["speed"].to(device, non_blocking=True),
        "hrr": batch["hrr"].to(device, non_blocking=True),
        "position": batch["position"].to(device, non_blocking=True),
        "speed_value": batch["speed_value"].to(device, non_blocking=True),
        "hrr_value": batch["hrr_value"].to(device, non_blocking=True),
    }
    if not getattr(model, "uses_image", True):
        return model.forward_from_image_features(image_features=None, **kwargs)
    return model(image=batch["image"].to(device, non_blocking=True), **kwargs)


def empty_totals() -> dict[str, float]:
    return {
        "samples": 0.0,
        "elements": 0.0,
        "absolute_error": 0.0,
        "squared_error": 0.0,
        "target_sum": 0.0,
        "target_squared_sum": 0.0,
        "standardized_squared_error": 0.0,
    }


def update_totals(
    totals: dict[str, float],
    prediction_raw: np.ndarray,
    target_raw: np.ndarray,
    prediction_standardized: torch.Tensor,
    target_standardized: torch.Tensor,
) -> None:
    error = prediction_raw - target_raw
    standardized_error = prediction_standardized.detach().cpu() - target_standardized.detach().cpu()
    totals["samples"] += float(prediction_raw.shape[0])
    totals["elements"] += float(prediction_raw.size)
    totals["absolute_error"] += float(np.abs(error).sum())
    totals["squared_error"] += float(np.square(error).sum())
    totals["target_sum"] += float(target_raw.sum())
    totals["target_squared_sum"] += float(np.square(target_raw).sum())
    totals["standardized_squared_error"] += float((standardized_error * standardized_error).sum().item())


def finalize_metrics(totals: dict[str, float]) -> dict[str, float]:
    samples = max(1.0, totals["samples"])
    elements = max(1.0, totals["elements"])
    mse = totals["squared_error"] / elements
    standardized_mse = totals["standardized_squared_error"] / elements
    total_sum_of_squares = totals["target_squared_sum"] - totals["target_sum"] ** 2 / elements
    r2 = float("nan")
    if total_sum_of_squares > 0:
        r2 = 1.0 - totals["squared_error"] / total_sum_of_squares
    return {
        "samples": int(samples),
        "loss": standardized_mse,
        "mae": totals["absolute_error"] / elements,
        "mse": mse,
        "rmse": math.sqrt(mse),
        "r2": r2,
    }


def row_from_prediction(record: Any, prediction: np.ndarray) -> dict[str, float]:
    values: dict[str, float] = {
        "e": float(record.env),
        "v": float(record.speed_value if record.speed_value is not None else record.speed),
        "q": float(record.hrr_value if record.hrr_value is not None else record.hrr),
        "s": float(record.position),
    }
    flat_prediction = prediction.reshape(-1)
    values.update(
        {
            column: float(value)
            for column, value in zip(TEMPERATURE_COLUMNS, flat_prediction, strict=True)
        }
    )
    return values


def predict_split(
    model: MultiModalTransformer,
    config: dict[str, Any],
    split_path: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    save_excel_path: Path | None = None,
) -> dict[str, float]:
    dataset = TrainFireDataset.from_config(config, split_path=split_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    rows_by_sheet: dict[str, list[tuple[int, dict[str, float]]]] = {}
    totals = empty_totals()
    cursor = 0
    total_batches = len(loader)
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            prediction = forward_batch(model, batch, device)
            prediction_raw = inverse_standardize_temperature(
                prediction.detach().cpu().numpy(),
                config,
            )
            target_raw = batch["target_raw"].numpy()
            update_totals(totals, prediction_raw, target_raw, prediction, batch["target"])

            batch_size_actual = int(prediction_raw.shape[0])
            if save_excel_path is not None:
                records = dataset.records[cursor : cursor + batch_size_actual]
                for record, sample_prediction in zip(records, prediction_raw, strict=True):
                    rows_by_sheet.setdefault(record.sheet_name, []).append(
                        (record.row_index, row_from_prediction(record, sample_prediction))
                    )
            cursor += batch_size_actual

            if batch_index == 1 or batch_index == total_batches or batch_index % 20 == 0:
                print(
                    f"{split_path.name}: batch {batch_index}/{total_batches} "
                    f"({cursor}/{len(dataset)} samples)",
                    flush=True,
                )

    if save_excel_path is not None:
        save_excel_path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(save_excel_path, engine="openpyxl") as writer:
            for sheet_name in sorted(rows_by_sheet):
                sheet_rows = [row for _, row in sorted(rows_by_sheet[sheet_name], key=lambda item: item[0])]
                df = pd.DataFrame(sheet_rows, columns=PHYSICAL_COLUMNS + TEMPERATURE_COLUMNS)
                df.to_excel(writer, sheet_name=sheet_name, index=False)

    return finalize_metrics(totals)


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "split", "checkpoint", "split_path", "samples", "loss", "mae", "mse", "rmse", "r2"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-dir", default="outputs/requested_predictions_20260622")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    torch.set_grad_enabled(False)

    image_only_checkpoint = Path(
        "outputs/exp_cmp_image_only_vit_b16_tunnel_random622_to_open_full_100ep/checkpoints/best.pt"
    )
    main_checkpoint = Path("outputs/exp_tunnel_random622_to_open_full_100ep/checkpoints/best.pt")
    tunnel_split = Path("data/splits/exp_tunnel_random_622_test.csv")
    open_split = Path("data/splits/exp_open_full_test.csv")

    metrics_rows: list[dict[str, Any]] = []

    print(f"Using device: {device}", flush=True)
    print("Evaluating image-only ViT on tunnel random622 test...", flush=True)
    image_model, image_config = load_checkpoint_model(image_only_checkpoint, device)
    image_metrics = predict_split(
        image_model,
        image_config,
        tunnel_split,
        device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    metrics_rows.append(
        {
            "model": "image_only_vit_b16",
            "split": "tunnel_random622_test",
            "checkpoint": str(image_only_checkpoint),
            "split_path": str(tunnel_split),
            **image_metrics,
        }
    )
    del image_model

    print("Running main model predictions for tunnel/open test splits...", flush=True)
    main_model, main_config = load_checkpoint_model(main_checkpoint, device)
    for split_name, split_path, excel_name in [
        ("tunnel_random622_test", tunnel_split, "main_model_tunnel_random622_test_predictions.xlsx"),
        ("open_full_test", open_split, "main_model_open_full_test_predictions.xlsx"),
    ]:
        excel_path = output_dir / excel_name
        split_metrics = predict_split(
            main_model,
            main_config,
            split_path,
            device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            save_excel_path=excel_path,
        )
        metrics_rows.append(
            {
                "model": "main_vit_physics",
                "split": split_name,
                "checkpoint": str(main_checkpoint),
                "split_path": str(split_path),
                **split_metrics,
            }
        )
        print(f"Wrote {excel_path}", flush=True)

    metrics_csv = output_dir / "metrics.csv"
    metrics_json = output_dir / "metrics.json"
    write_metrics_csv(metrics_csv, metrics_rows)
    metrics_json.write_text(json.dumps(metrics_rows, indent=2), encoding="utf-8")
    print(f"Wrote {metrics_csv}", flush=True)
    print(f"Wrote {metrics_json}", flush=True)


if __name__ == "__main__":
    main()
