from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from move_train.train import train, write_split_metrics


def _make_training_config(tmp_path: Path) -> dict:
    data_root = tmp_path / "data"
    sheet = "1-1"
    env = 0
    run_dir = data_root / str(env) / f"{env}-{sheet}"
    run_dir.mkdir(parents=True)
    for frame in range(1, 7):
        Image.new("RGB", (32, 32), color=(frame * 20, 0, 0)).save(
            run_dir / f"{env}-{sheet} ({frame}).png"
        )

    rows = []
    for frame in range(1, 7):
        row = {"e": env, "v": 1, "q": 1, "s": frame / 10}
        row.update({f"T{i}": float(i + frame) for i in range(1, 46)})
        rows.append(row)
    excel_path = tmp_path / "data_0.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name=sheet, index=False)

    return {
        "data": {
            "root": str(data_root),
            "excel_files": {env: str(excel_path)},
            "environments": [env],
            "image_size": [32, 32],
            "image_mean": [0.0, 0.0, 0.0],
            "image_std": [1.0, 1.0, 1.0],
            "position_scale": 10.0,
            "val_fraction": 0.33,
            "random_seed": 7,
            "num_workers": 0,
            "pin_memory": False,
            "target_standardization": {
                "enabled": True,
                "mode": "per_sensor",
                "epsilon": 1e-6,
            },
        },
        "model": {
            "image_size": [32, 32],
            "image_backbone": "custom",
            "pretrained_backbone": False,
            "patch_size": 8,
            "in_channels": 3,
            "embed_dim": 32,
            "image_depth": 1,
            "physical_depth": 1,
            "fusion_depth": 1,
            "num_heads": 4,
            "mlp_ratio": 2.0,
            "dropout": 0.0,
            "num_envs": 2,
            "num_speeds": 8,
            "num_hrr": 7,
            "use_continuous_physics": True,
            "speed_scale": 7.0,
            "hrr_scale": 6.0,
            "output_rows": 5,
            "output_cols": 9,
        },
        "train": {
            "batch_size": 2,
            "epochs": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "output_dir": str(tmp_path / "outputs"),
            "checkpoint_dir": str(tmp_path / "outputs" / "checkpoints"),
            "amp": False,
            "max_train_batches": 1,
            "max_val_batches": 1,
        },
    }


def test_train_writes_metrics_and_checkpoints(tmp_path: Path) -> None:
    result = train(_make_training_config(tmp_path), device_name="cpu")

    assert result["metrics_path"].exists()
    assert result["test_metrics_path"] is None
    metrics = pd.read_csv(result["metrics_path"])
    assert "train_r2" in metrics.columns
    assert "val_r2" in metrics.columns
    assert (result["checkpoint_dir"] / "best.pt").exists()
    assert (result["checkpoint_dir"] / "last.pt").exists()
    checkpoint = torch.load(
        result["checkpoint_dir"] / "best.pt",
        map_location="cpu",
        weights_only=False,
    )
    target_cfg = checkpoint["config"]["data"]["target_standardization"]
    assert target_cfg["fitted_on"] == "train_split"
    assert len(target_cfg["mean"]) == 5
    assert len(target_cfg["std"][0]) == 9
    assert "r2" in checkpoint["metrics"]
    assert "scheduler_state_dict" in checkpoint


def test_write_split_metrics_includes_r2(tmp_path: Path) -> None:
    path = tmp_path / "test_metrics.csv"
    write_split_metrics(
        path,
        "test",
        {
            "loss": 1.0,
            "mae": 2.0,
            "mse": 3.0,
            "rmse": 4.0,
            "r2": 0.5,
        },
    )

    metrics = pd.read_csv(path)
    assert list(metrics.columns) == ["split", "loss", "mae", "mse", "rmse", "r2"]
    assert metrics.loc[0, "split"] == "test"
    assert metrics.loc[0, "r2"] == 0.5
