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
    assert sample["position"].item() == pytest.approx(0.01)


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
