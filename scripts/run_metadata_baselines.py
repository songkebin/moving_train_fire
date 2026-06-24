from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from move_train.config import load_config
from move_train.data import SampleRecord, build_records


BASE_CONFIG = "configs/exp_tunnel_random622_to_open_full_100ep.yaml"
OUTPUT_DIR = Path("outputs/exp_cmp_metadata_sklearn")

SPLITS = {
    "tunnel_train": "data/splits/exp_tunnel_random_622_train.csv",
    "open_full": "data/splits/exp_open_full_test.csv",
    "open_random20_finetune": "data/splits/exp_open_random20_finetune.csv",
    "open_random20_test": "data/splits/exp_open_random20_test.csv",
    "open_condition20_finetune": "data/splits/exp_open_condition20_finetune.csv",
    "open_condition20_test": "data/splits/exp_open_condition20_test.csv",
}

PROTOCOLS = {
    "direct_full": {
        "train": ["tunnel_train"],
        "test": "open_full",
        "description": "Train on tunnel random 6:2:2 train; test on full open split.",
    },
    "random20_source_plus_target": {
        "train": ["tunnel_train", "open_random20_finetune"],
        "test": "open_random20_test",
        "description": "Train on tunnel train plus random 20% open samples; test on remaining open samples.",
    },
    "random20_target_only": {
        "train": ["open_random20_finetune"],
        "test": "open_random20_test",
        "description": "Train only on random 20% open samples; test on remaining open samples.",
    },
    "condition20_source_plus_target": {
        "train": ["tunnel_train", "open_condition20_finetune"],
        "test": "open_condition20_test",
        "description": "Train on tunnel train plus selected open conditions; test on unseen open conditions.",
    },
    "condition20_target_only": {
        "train": ["open_condition20_finetune"],
        "test": "open_condition20_test",
        "description": "Train only on selected open conditions; test on unseen open conditions.",
    },
}


def features(records: Iterable[SampleRecord]) -> np.ndarray:
    rows = []
    for record in records:
        speed = record.speed_value if record.speed_value is not None else float(record.speed)
        hrr = record.hrr_value if record.hrr_value is not None else float(record.hrr)
        position = float(record.position)
        env = float(record.env)
        rows.append([env, float(speed), float(hrr), position])
    return np.asarray(rows, dtype=np.float32)


def targets(records: Iterable[SampleRecord]) -> np.ndarray:
    return np.asarray([record.target.reshape(-1) for record in records], dtype=np.float32)


def compute_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = prediction - target
    mse = float(np.mean(error * error))
    target_mean = float(np.mean(target))
    total_sum_of_squares = float(np.sum((target - target_mean) ** 2))
    squared_error = float(np.sum(error * error))
    r2 = float("nan")
    if total_sum_of_squares > 0.0:
        r2 = 1.0 - squared_error / total_sum_of_squares
    return {
        "mae": float(np.mean(np.abs(error))),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": r2,
    }


def make_models() -> dict:
    return {
        "metadata_ridge_poly3": make_pipeline(
            PolynomialFeatures(degree=3, include_bias=False),
            StandardScaler(),
            RidgeCV(alphas=np.logspace(-4, 4, 17)),
        ),
        "metadata_extra_trees": ExtraTreesRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=1,
        ),
        "metadata_random_forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=1,
        ),
    }


def main() -> None:
    config = load_config(BASE_CONFIG)
    records_by_split = {
        name: build_records(config, split_path=path) for name, path in SPLITS.items()
    }

    run_dir = OUTPUT_DIR / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "metadata_baseline_metrics.csv"

    rows = []
    for protocol_name, protocol in PROTOCOLS.items():
        train_records = []
        for split_name in protocol["train"]:
            train_records.extend(records_by_split[split_name])
        test_records = records_by_split[protocol["test"]]
        x_train = features(train_records)
        y_train = targets(train_records)
        x_test = features(test_records)
        y_test = targets(test_records)

        for model_name, model in make_models().items():
            model.fit(x_train, y_train)
            prediction = model.predict(x_test)
            metrics = compute_metrics(prediction, y_test)
            row = {
                "protocol": protocol_name,
                "model": model_name,
                "train_splits": "+".join(protocol["train"]),
                "test_split": protocol["test"],
                "train_samples": len(train_records),
                "test_samples": len(test_records),
                **metrics,
            }
            rows.append(row)
            print(
                f"{protocol_name} {model_name}: "
                f"rmse={metrics['rmse']:.4f} mae={metrics['mae']:.4f} r2={metrics['r2']:.4f}"
            )

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
