from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch

from move_train.config import load_config
from move_train.data import TrainFireDataset, inverse_standardize_temperature, load_image_tensor
from move_train.model import MultiModalTransformer


def load_model(checkpoint_path: str | Path, config: dict[str, Any], device: torch.device) -> MultiModalTransformer:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_config = checkpoint.get("config")
    if checkpoint_config is not None:
        config.clear()
        config.update(checkpoint_config)
    model_config = config["model"]
    model = MultiModalTransformer(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def save_temperature_csv(path: Path, temperature: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in temperature:
            writer.writerow([f"{float(value):.6f}" for value in row])


def save_temperature_png(path: Path, temperature: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
    image = ax.imshow(temperature, cmap="inferno", aspect="auto")
    ax.set_xlabel("Thermocouple column")
    ax.set_ylabel("Thermocouple row")
    fig.colorbar(image, ax=ax, label="Temperature")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def predict_single(
    model: MultiModalTransformer,
    config: dict[str, Any],
    device: torch.device,
    image_path: str | Path,
    env: int,
    speed: int,
    hrr: int,
    position: float,
    speed_value: float | None = None,
    hrr_value: float | None = None,
) -> np.ndarray:
    data_cfg = config["data"]
    image = load_image_tensor(
        image_path,
        image_size=tuple(data_cfg.get("image_size", (224, 224))),
        image_mean=torch.tensor(data_cfg.get("image_mean", (0.485, 0.456, 0.406)), dtype=torch.float32).view(3, 1, 1),
        image_std=torch.tensor(data_cfg.get("image_std", (0.229, 0.224, 0.225)), dtype=torch.float32).view(3, 1, 1),
    ).unsqueeze(0)
    position_scale = float(data_cfg.get("position_scale", 11.0))
    batch = {
        "image": image.to(device),
        "env": torch.tensor([env], dtype=torch.long, device=device),
        "speed": torch.tensor([speed], dtype=torch.long, device=device),
        "hrr": torch.tensor([hrr], dtype=torch.long, device=device),
        "speed_value": torch.tensor(
            [speed if speed_value is None else speed_value], dtype=torch.float32, device=device
        ),
        "hrr_value": torch.tensor(
            [hrr if hrr_value is None else hrr_value], dtype=torch.float32, device=device
        ),
        "position": torch.tensor([position / position_scale], dtype=torch.float32, device=device),
    }
    with torch.no_grad():
        prediction = model(**batch)
    prediction_array = prediction.squeeze(0).detach().cpu().numpy()
    return inverse_standardize_temperature(prediction_array, config)


def predict_dataset(
    model: MultiModalTransformer,
    config: dict[str, Any],
    device: torch.device,
    output_dir: Path,
    limit: int | None = None,
) -> None:
    dataset = TrainFireDataset.from_config(config)
    count = len(dataset) if limit is None else min(limit, len(dataset))
    for index in range(count):
        sample = dataset[index]
        batch = {
            "image": sample["image"].unsqueeze(0).to(device),
            "env": sample["env"].view(1).to(device),
            "speed": sample["speed"].view(1).to(device),
            "hrr": sample["hrr"].view(1).to(device),
            "speed_value": sample["speed_value"].view(1).to(device),
            "hrr_value": sample["hrr_value"].view(1).to(device),
            "position": sample["position"].view(1).to(device),
        }
        with torch.no_grad():
            prediction = model(**batch).squeeze(0).cpu().numpy()
        prediction = inverse_standardize_temperature(prediction, config)
        stem = f"{index:06d}_{Path(sample['metadata']['image_path']).stem}"
        save_temperature_csv(output_dir / f"{stem}.csv", prediction)
        save_temperature_png(output_dir / f"{stem}.png", prediction)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run temperature-field prediction.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument("--device", default=None, help="Torch device override, e.g. cuda or cpu.")
    parser.add_argument("--output-dir", default="outputs/predictions", help="Directory for prediction files.")
    parser.add_argument("--limit", type=int, default=None, help="Limit dataset predictions.")
    parser.add_argument("--image", default=None, help="Optional single image path.")
    parser.add_argument("--env", type=int, default=None, help="Environment token for single-image prediction.")
    parser.add_argument("--speed", type=int, default=None, help="Speed category ID for single-image prediction.")
    parser.add_argument("--hrr", type=int, default=None, help="HRR category ID for single-image prediction.")
    parser.add_argument("--speed-value", type=float, default=None, help="Physical speed value for single-image prediction.")
    parser.add_argument("--hrr-value", type=float, default=None, help="Physical HRR value for single-image prediction.")
    parser.add_argument("--position", type=float, default=None, help="Position value for single-image prediction.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(args.checkpoint, config, device)
    output_dir = Path(args.output_dir)

    single_values = [args.image, args.env, args.speed, args.hrr, args.position]
    if any(value is not None for value in single_values):
        if any(value is None for value in single_values):
            raise ValueError("--image, --env, --speed, --hrr, and --position must be provided together.")
        prediction = predict_single(
            model,
            config,
            device,
            image_path=args.image,
            env=args.env,
            speed=args.speed,
            hrr=args.hrr,
            position=args.position,
            speed_value=args.speed_value,
            hrr_value=args.hrr_value,
        )
        save_temperature_csv(output_dir / "single_prediction.csv", prediction)
        save_temperature_png(output_dir / "single_prediction.png", prediction)
    else:
        predict_dataset(model, config, device, output_dir=output_dir, limit=args.limit)


if __name__ == "__main__":
    main()
