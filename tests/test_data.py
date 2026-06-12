from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from move_train.data import DataAlignmentError, TrainFireDataset, build_records


def _temperature_row(env: int, speed: int, hrr: int, position: float) -> dict[str, float]:
    row: dict[str, float] = {"e": env, "v": speed, "q": hrr, "s": position}
    for index in range(1, 46):
        row[f"T{index}"] = float(index)
    return row


def _write_fixture(
    tmp_path: Path,
    env: int = 0,
    sheet: str = "1-1",
    rows: int = 3,
    frame_numbers: tuple[int, ...] = (1, 2, 3),
) -> dict:
    data_root = tmp_path / "data"
    run_dir = data_root / str(env) / f"{env}-{sheet}"
    run_dir.mkdir(parents=True)
    for frame in frame_numbers:
        Image.new("RGB", (12, 8), color=(frame * 20 % 255, 0, 0)).save(
            run_dir / f"{env}-{sheet} ({frame}).png"
        )

    excel_path = tmp_path / f"data_{env}.xlsx"
    df = pd.DataFrame(
        [_temperature_row(env, speed=1, hrr=1, position=(idx + 1) / 10) for idx in range(rows)]
    )
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet, index=False)

    return {
        "data": {
            "root": str(data_root),
            "excel_files": {env: str(excel_path)},
            "environments": [env],
            "image_size": [16, 16],
            "image_mean": [0.0, 0.0, 0.0],
            "image_std": [1.0, 1.0, 1.0],
            "position_scale": 10.0,
        }
    }


def test_build_records_sorts_images_by_frame_number(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path, frame_numbers=(10, 1, 2), rows=3)

    records = build_records(config)

    assert [record.frame_index for record in records] == [1, 2, 10]
    assert records[0].image_path.name.endswith("(1).png")


def test_dataset_returns_expected_tensor_shapes(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path)

    dataset = TrainFireDataset.from_config(config)
    sample = dataset[0]

    assert sample["image"].shape == (3, 16, 16)
    assert sample["target"].shape == (5, 9)
    assert sample["env"].item() == 0
    assert sample["speed"].item() == 1
    assert sample["hrr"].item() == 1
    assert sample["speed_value"].item() == pytest.approx(1.0)
    assert sample["hrr_value"].item() == pytest.approx(1.0)
    assert sample["position"].item() == pytest.approx(0.01)


def test_manifest_split_records_use_category_ids(tmp_path: Path) -> None:
    image_path = tmp_path / "raw" / "deceleration" / "0-1.5-1" / "0-1.5-1 (1).png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (12, 8), color=(100, 0, 0)).save(image_path)

    manifest_path = tmp_path / "samples.csv"
    split_path = tmp_path / "split.csv"
    row = {
        "sample_id": "deceleration_0-1.5-1_f001",
        "run_id": "deceleration_0-1.5-1",
        "source": "deceleration",
        "motion_regime": "deceleration",
        "environment": "tunnel",
        "environment_id": 0,
        "image_path": str(image_path),
        "table_path": str(tmp_path / "deceleration.xlsx"),
        "sheet_name": "0-1.5-1",
        "row_index": 0,
        "frame_index": 1,
        "speed": 1.5,
        "speed_id": 2,
        "hrr": 4,
        "hrr_id": 3,
        "position": 0.1,
        "alignment_status": "ok",
    }
    row.update({f"T{i}": float(i) for i in range(1, 46)})
    pd.DataFrame([row]).to_csv(manifest_path, index=False)
    pd.DataFrame([{"sample_id": row["sample_id"]}]).to_csv(split_path, index=False)

    config = {
        "data": {
            "root": str(tmp_path),
            "manifest": str(manifest_path),
            "image_size": [16, 16],
            "image_mean": [0.0, 0.0, 0.0],
            "image_std": [1.0, 1.0, 1.0],
            "position_scale": 10.0,
        }
    }

    records = build_records(config, split_path=split_path)
    dataset = TrainFireDataset.from_config(config, split_path=split_path)
    sample = dataset[0]

    assert len(records) == 1
    assert records[0].speed == 2
    assert records[0].speed_value == 1.5
    assert records[0].hrr_value == 4.0
    assert sample["speed"].item() == 2
    assert sample["hrr"].item() == 3
    assert sample["speed_value"].item() == pytest.approx(1.5)
    assert sample["hrr_value"].item() == pytest.approx(4.0)
    assert sample["metadata"]["motion_regime"] == "deceleration"


def test_mismatch_between_rows_and_frames_raises(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path, rows=4, frame_numbers=(1, 2, 3))

    with pytest.raises(DataAlignmentError, match="rows=4.*frames=3"):
        build_records(config)


def test_missing_physical_values_raise_alignment_error(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path, rows=3, frame_numbers=(1, 2, 3))
    excel_path = Path(config["data"]["excel_files"][0])
    df = pd.read_excel(excel_path, sheet_name="1-1")
    df.loc[2, "v"] = None
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="1-1", index=False)

    with pytest.raises(DataAlignmentError, match=r"row_index=2 \(Excel row 4\): v"):
        build_records(config)
