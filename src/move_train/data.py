from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, random_split

TEMPERATURE_COLUMNS = [f"T{i}" for i in range(1, 46)]
PHYSICAL_COLUMNS = ["e", "v", "q", "s"]
FRAME_RE = re.compile(r"\((\d+)\)\.png$", re.IGNORECASE)


class DataAlignmentError(ValueError):
    """Raised when spreadsheet rows and image frames cannot be aligned."""


@dataclass(frozen=True)
class SampleRecord:
    image_path: Path
    env: int
    speed: int
    hrr: int
    position: float
    target: np.ndarray
    excel_path: Path
    sheet_name: str
    row_index: int
    frame_index: int
    sample_id: str | None = None
    run_id: str | None = None
    source: str | None = None
    motion_regime: str | None = None
    environment_name: str | None = None
    speed_value: float | None = None
    hrr_value: float | None = None


def _as_path(path: str | Path, base: Path | None = None) -> Path:
    p = Path(path)
    if not p.is_absolute() and base is not None:
        p = base / p
    return p


def _frame_index(path: Path) -> int:
    match = FRAME_RE.search(path.name)
    if match is None:
        raise DataAlignmentError(f"Cannot parse frame number from image name: {path}")
    return int(match.group(1))


def _sorted_frames(run_dir: Path) -> list[Path]:
    frames = list(run_dir.glob("*.png"))
    if not frames:
        raise DataAlignmentError(f"No PNG frames found in image directory: {run_dir}")
    return sorted(frames, key=_frame_index)


def _read_sheet(excel_path: Path, sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(excel_path, sheet_name=sheet_name, engine="openpyxl")
    df = df.dropna(how="all").copy()
    df.columns = [str(col).strip() for col in df.columns]

    required = set(PHYSICAL_COLUMNS + TEMPERATURE_COLUMNS)
    missing = sorted(required.difference(df.columns))
    if missing:
        raise DataAlignmentError(
            f"{excel_path} sheet {sheet_name!r} is missing required columns: {missing}"
        )
    selected = df[PHYSICAL_COLUMNS + TEMPERATURE_COLUMNS].copy()
    for column in selected.columns:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    return selected


def _missing_values_message(excel_path: Path, sheet_name: str, df: pd.DataFrame) -> str | None:
    missing_mask = df.isna()
    bad_rows = missing_mask.any(axis=1)
    if not bad_rows.any():
        return None

    examples: list[str] = []
    for row_index in df.index[bad_rows][:5]:
        columns = [str(column) for column in df.columns[missing_mask.loc[row_index]]]
        excel_row = int(row_index) + 2
        examples.append(
            f"row_index={row_index} (Excel row {excel_row}): {','.join(columns)}"
        )
    suffix = "" if bad_rows.sum() <= 5 else f"; plus {int(bad_rows.sum()) - 5} more"
    return (
        f"{excel_path} sheet {sheet_name!r} contains missing/non-numeric values "
        f"({'; '.join(examples)}{suffix})"
    )


def _records_for_excel(data_root: Path, excel_path: Path, environment: int) -> list[SampleRecord]:
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    workbook = pd.ExcelFile(excel_path, engine="openpyxl")
    records: list[SampleRecord] = []
    errors: list[str] = []
    for sheet_name in workbook.sheet_names:
        run_name = f"{environment}-{sheet_name}"
        run_dir = data_root / str(environment) / run_name
        try:
            df = _read_sheet(excel_path, sheet_name)
        except DataAlignmentError as exc:
            errors.append(str(exc))
            continue

        sheet_errors: list[str] = []
        try:
            frames = _sorted_frames(run_dir)
        except DataAlignmentError as exc:
            sheet_errors.append(str(exc))
            frames = []

        if len(df) != len(frames):
            sheet_errors.append(
                "Spreadsheet/image count mismatch: "
                f"excel={excel_path}, sheet={sheet_name}, rows={len(df)}, "
                f"image_dir={run_dir}, frames={len(frames)}"
            )

        missing_message = _missing_values_message(excel_path, sheet_name, df)
        if missing_message is not None:
            sheet_errors.append(missing_message)

        if sheet_errors:
            errors.extend(sheet_errors)
            continue

        for row_idx, (row, image_path) in enumerate(zip(df.itertuples(index=False), frames)):
            values = row._asdict()
            target = np.asarray(
                [float(values[col]) for col in TEMPERATURE_COLUMNS], dtype=np.float32
            ).reshape(5, 9)
            records.append(
                SampleRecord(
                    image_path=image_path,
                    env=int(values["e"]),
                    speed=int(values["v"]),
                    hrr=int(values["q"]),
                    position=float(values["s"]),
                    target=target,
                    excel_path=excel_path,
                    sheet_name=sheet_name,
                    row_index=row_idx,
                    frame_index=_frame_index(image_path),
                )
            )
    if errors:
        raise DataAlignmentError("Data validation failed:\n- " + "\n- ".join(errors))
    return records


def _split_sample_ids(split_path: Path | None) -> set[str] | None:
    if split_path is None:
        return None
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    split_df = pd.read_csv(split_path)
    if "sample_id" not in split_df.columns:
        raise DataAlignmentError(f"Split file is missing sample_id column: {split_path}")
    return set(split_df["sample_id"].astype(str))


def _records_for_manifest(
    project_root: Path,
    manifest_path: Path,
    split_path: Path | None = None,
) -> list[SampleRecord]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    required = {
        "sample_id",
        "image_path",
        "table_path",
        "sheet_name",
        "row_index",
        "frame_index",
        "environment_id",
        "speed_id",
        "hrr_id",
        "position",
        *TEMPERATURE_COLUMNS,
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise DataAlignmentError(f"{manifest_path} is missing required columns: {missing}")

    sample_ids = _split_sample_ids(split_path)
    if sample_ids is not None:
        df = df[df["sample_id"].astype(str).isin(sample_ids)].copy()
        discovered = set(df["sample_id"].astype(str))
        missing_ids = sorted(sample_ids.difference(discovered))
        if missing_ids:
            examples = ", ".join(missing_ids[:5])
            suffix = "" if len(missing_ids) <= 5 else f", plus {len(missing_ids) - 5} more"
            raise DataAlignmentError(
                f"{split_path} references sample_id values missing from {manifest_path}: "
                f"{examples}{suffix}"
            )

    numeric_columns = [
        "row_index",
        "frame_index",
        "environment_id",
        "speed_id",
        "hrr_id",
        "position",
        *TEMPERATURE_COLUMNS,
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    missing_message = _missing_values_message(manifest_path, manifest_path.name, df[numeric_columns])
    if missing_message is not None:
        raise DataAlignmentError(missing_message)

    records: list[SampleRecord] = []
    for row in df.itertuples(index=False):
        values = row._asdict()
        image_path = _as_path(str(values["image_path"]), project_root)
        if not image_path.exists():
            raise DataAlignmentError(f"Manifest image path does not exist: {image_path}")
        target = np.asarray(
            [float(values[col]) for col in TEMPERATURE_COLUMNS], dtype=np.float32
        ).reshape(5, 9)
        records.append(
            SampleRecord(
                image_path=image_path,
                env=int(values["environment_id"]),
                speed=int(values["speed_id"]),
                hrr=int(values["hrr_id"]),
                position=float(values["position"]),
                target=target,
                excel_path=_as_path(str(values["table_path"]), project_root),
                sheet_name=str(values["sheet_name"]),
                row_index=int(values["row_index"]),
                frame_index=int(values["frame_index"]),
                sample_id=str(values["sample_id"]),
                run_id=str(values["run_id"]) if "run_id" in values else None,
                source=str(values["source"]) if "source" in values else None,
                motion_regime=str(values["motion_regime"]) if "motion_regime" in values else None,
                environment_name=str(values["environment"]) if "environment" in values else None,
                speed_value=float(values["speed"]) if "speed" in values else None,
                hrr_value=float(values["hrr"]) if "hrr" in values else None,
            )
        )
    return records


def build_records(config: dict[str, Any], split_path: str | Path | None = None) -> list[SampleRecord]:
    data_cfg = config["data"]
    project_root = Path.cwd()
    if "manifest" in data_cfg:
        manifest_path = _as_path(data_cfg["manifest"], project_root)
        split = _as_path(split_path, project_root) if split_path is not None else None
        records = _records_for_manifest(project_root, manifest_path, split)
        if not records:
            raise DataAlignmentError("No samples were discovered from the configured manifest/split.")
        return records

    data_root = _as_path(data_cfg["root"], project_root)
    excel_files = data_cfg["excel_files"]
    environments: Iterable[int] = data_cfg.get("environments", sorted(map(int, excel_files)))

    records: list[SampleRecord] = []
    errors: list[str] = []
    for environment in environments:
        key = str(environment)
        excel_value = excel_files.get(key, excel_files.get(environment))
        if excel_value is None:
            raise KeyError(f"No Excel file configured for environment {environment}")
        try:
            records.extend(
                _records_for_excel(data_root, _as_path(excel_value, project_root), environment)
            )
        except DataAlignmentError as exc:
            errors.append(str(exc))

    if errors:
        raise DataAlignmentError(
            "Data validation failed across configured sources:\n" + "\n".join(errors)
        )

    if not records:
        raise DataAlignmentError("No samples were discovered from the configured data sources.")
    return records


class TrainFireDataset(Dataset):
    def __init__(
        self,
        records: list[SampleRecord],
        image_size: tuple[int, int] = (224, 224),
        image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        image_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        position_scale: float = 11.0,
    ) -> None:
        self.records = records
        self.image_size = tuple(image_size)
        self.image_mean = torch.tensor(image_mean, dtype=torch.float32).view(3, 1, 1)
        self.image_std = torch.tensor(image_std, dtype=torch.float32).view(3, 1, 1)
        self.position_scale = float(position_scale)
        self.target_mean: torch.Tensor | None = None
        self.target_std: torch.Tensor | None = None

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        split_path: str | Path | None = None,
    ) -> "TrainFireDataset":
        data_cfg = config["data"]
        dataset = cls(
            records=build_records(config, split_path=split_path),
            image_size=tuple(data_cfg.get("image_size", (224, 224))),
            image_mean=tuple(data_cfg.get("image_mean", (0.485, 0.456, 0.406))),
            image_std=tuple(data_cfg.get("image_std", (0.229, 0.224, 0.225))),
            position_scale=float(data_cfg.get("position_scale", 11.0)),
        )
        target_cfg = data_cfg.get("target_standardization", {})
        if target_cfg.get("enabled", False) and "mean" in target_cfg and "std" in target_cfg:
            dataset.set_target_standardization(
                np.asarray(target_cfg["mean"], dtype=np.float32),
                np.asarray(target_cfg["std"], dtype=np.float32),
            )
        return dataset

    def set_target_standardization(self, mean: np.ndarray, std: np.ndarray) -> None:
        mean_array = np.asarray(mean, dtype=np.float32).reshape(5, 9)
        std_array = np.asarray(std, dtype=np.float32).reshape(5, 9)
        if np.any(std_array <= 0):
            raise ValueError("All target standard deviations must be positive.")
        self.target_mean = torch.from_numpy(mean_array)
        self.target_std = torch.from_numpy(std_array)

    @property
    def has_target_standardization(self) -> bool:
        return self.target_mean is not None and self.target_std is not None

    def standardize_target(self, target: torch.Tensor) -> torch.Tensor:
        if not self.has_target_standardization:
            return target
        assert self.target_mean is not None and self.target_std is not None
        return (target - self.target_mean) / self.target_std

    def inverse_standardize_target(self, target: torch.Tensor) -> torch.Tensor:
        if not self.has_target_standardization:
            return target
        assert self.target_mean is not None and self.target_std is not None
        mean = self.target_mean.to(device=target.device, dtype=target.dtype)
        std = self.target_std.to(device=target.device, dtype=target.dtype)
        return target * std + mean

    def __len__(self) -> int:
        return len(self.records)

    def sample_without_image(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        target_raw = torch.from_numpy(record.target)
        speed_value = record.speed_value if record.speed_value is not None else float(record.speed)
        hrr_value = record.hrr_value if record.hrr_value is not None else float(record.hrr)
        return {
            "env": torch.tensor(record.env, dtype=torch.long),
            "speed": torch.tensor(record.speed, dtype=torch.long),
            "hrr": torch.tensor(record.hrr, dtype=torch.long),
            "speed_value": torch.tensor(speed_value, dtype=torch.float32),
            "hrr_value": torch.tensor(hrr_value, dtype=torch.float32),
            "position": torch.tensor(record.position / self.position_scale, dtype=torch.float32),
            "target": self.standardize_target(target_raw),
            "target_raw": target_raw,
            "metadata": {
                "image_path": str(record.image_path),
                "excel_path": str(record.excel_path),
                "sheet_name": record.sheet_name,
                "row_index": record.row_index,
                "frame_index": record.frame_index,
                "sample_id": record.sample_id or "",
                "run_id": record.run_id or "",
                "source": record.source or "",
                "motion_regime": record.motion_regime or "",
                "environment": record.environment_name or "",
                "speed_value": "" if record.speed_value is None else record.speed_value,
                "hrr_value": "" if record.hrr_value is None else record.hrr_value,
            },
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.sample_without_image(index)
        record = self.records[index]
        sample["image"] = load_image_tensor(
            record.image_path,
            image_size=self.image_size,
            image_mean=self.image_mean,
            image_std=self.image_std,
        )
        return sample


def load_image_tensor(
    image_path: str | Path,
    image_size: tuple[int, int],
    image_mean: torch.Tensor,
    image_std: torch.Tensor,
) -> torch.Tensor:
    height, width = image_size
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    with Image.open(image_path) as image:
        image = image.convert("RGB").resize((width, height), resampling)
        array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return (tensor - image_mean) / image_std


def split_dataset(dataset: Dataset, val_fraction: float, seed: int) -> tuple[Dataset, Dataset]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    val_size = max(1, int(round(len(dataset) * val_fraction)))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("Dataset is too small to create a non-empty training split.")
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def compute_target_standardization(
    records: list[SampleRecord],
    indices: Iterable[int],
    epsilon: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    targets = np.stack([records[int(index)].target for index in indices]).astype(np.float32)
    mean = targets.mean(axis=0)
    std = targets.std(axis=0)
    std = np.maximum(std, float(epsilon)).astype(np.float32)
    return mean.astype(np.float32), std


def target_standardization_from_config(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    target_cfg = config.get("data", {}).get("target_standardization", {})
    if not target_cfg.get("enabled", False):
        return None
    if "mean" not in target_cfg or "std" not in target_cfg:
        return None
    return (
        np.asarray(target_cfg["mean"], dtype=np.float32).reshape(5, 9),
        np.asarray(target_cfg["std"], dtype=np.float32).reshape(5, 9),
    )


def inverse_standardize_temperature(temperature: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    stats = target_standardization_from_config(config)
    if stats is None:
        return temperature
    mean, std = stats
    return temperature * std + mean
