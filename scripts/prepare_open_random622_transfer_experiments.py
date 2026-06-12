from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data/manifests/samples.csv"
SPLIT_DIR = PROJECT_ROOT / "data/splits"
CONFIG_DIR = PROJECT_ROOT / "configs"
SEED = 42


def write_split(path: Path, sample_ids: pd.Series) -> None:
    df = pd.DataFrame({"sample_id": sample_ids.astype(str).to_list()})
    df.to_csv(path, index=False)
    print(f"{path.relative_to(PROJECT_ROOT)}: {len(df)} samples")


def load_base_config() -> dict:
    with (CONFIG_DIR / "tunnel_condition_open_test.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_config(path: Path, config: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    print(path.relative_to(PROJECT_ROOT))


def main() -> None:
    manifest = pd.read_csv(MANIFEST_PATH)
    rng = np.random.default_rng(SEED)

    tunnel = manifest[
        (manifest["source"] == "constant_tunnel") & (manifest["environment"] == "tunnel")
    ].copy()
    open_segment = manifest[
        (manifest["source"] == "constant_open") & (manifest["environment"] == "open")
    ].copy()

    tunnel_indices = rng.permutation(len(tunnel))
    train_count = round(len(tunnel) * 0.6)
    val_count = round(len(tunnel) * 0.2)
    train_indices = tunnel_indices[:train_count]
    val_indices = tunnel_indices[train_count : train_count + val_count]
    test_indices = tunnel_indices[train_count + val_count :]
    write_split(
        SPLIT_DIR / "exp_tunnel_random_622_train.csv",
        tunnel.iloc[sorted(train_indices)]["sample_id"],
    )
    write_split(
        SPLIT_DIR / "exp_tunnel_random_622_val.csv",
        tunnel.iloc[sorted(val_indices)]["sample_id"],
    )
    write_split(
        SPLIT_DIR / "exp_tunnel_random_622_test.csv",
        tunnel.iloc[sorted(test_indices)]["sample_id"],
    )

    write_split(SPLIT_DIR / "exp_open_full_test.csv", open_segment["sample_id"])

    open_indices = np.arange(len(open_segment))
    random_finetune_indices = set(rng.permutation(open_indices)[: round(len(open_segment) * 0.2)])
    random_finetune = open_segment.iloc[sorted(random_finetune_indices)]
    random_test = open_segment.iloc[
        [index for index in range(len(open_segment)) if index not in random_finetune_indices]
    ]
    write_split(SPLIT_DIR / "exp_open_random20_finetune.csv", random_finetune["sample_id"])
    write_split(SPLIT_DIR / "exp_open_random20_test.csv", random_test["sample_id"])

    open_runs = sorted(open_segment["run_id"].unique())
    rng.shuffle(open_runs)
    condition_finetune_runs = set(open_runs[:8])
    condition_finetune = open_segment[open_segment["run_id"].isin(condition_finetune_runs)]
    condition_test = open_segment[~open_segment["run_id"].isin(condition_finetune_runs)]
    write_split(SPLIT_DIR / "exp_open_condition20_finetune.csv", condition_finetune["sample_id"])
    write_split(SPLIT_DIR / "exp_open_condition20_test.csv", condition_test["sample_id"])
    print(
        "open condition finetune runs "
        f"({len(condition_finetune_runs)}): {', '.join(sorted(condition_finetune_runs))}"
    )

    source_name = "exp_tunnel_random622_to_open_full_100ep"
    source_config = load_base_config()
    source_config["data"]["splits"]["train"] = "data/splits/exp_tunnel_random_622_train.csv"
    source_config["data"]["splits"]["val"] = "data/splits/exp_tunnel_random_622_val.csv"
    source_config["data"]["splits"]["test"] = "data/splits/exp_open_full_test.csv"
    source_config["train"]["epochs"] = 100
    source_config["train"]["output_dir"] = f"outputs/{source_name}"
    source_config["train"]["checkpoint_dir"] = f"outputs/{source_name}/checkpoints"
    source_config["train"]["cache_frozen_image_features"] = True
    write_config(CONFIG_DIR / f"{source_name}.yaml", source_config)

    finetune_configs = [
        (
            "exp_open_random20_finetune_from_tunnel_random622_100ep",
            "data/splits/exp_open_random20_finetune.csv",
            "data/splits/exp_open_random20_test.csv",
        ),
        (
            "exp_open_condition20_finetune_from_tunnel_random622_100ep",
            "data/splits/exp_open_condition20_finetune.csv",
            "data/splits/exp_open_condition20_test.csv",
        ),
    ]
    for name, train_split, test_split in finetune_configs:
        config = load_base_config()
        config["data"]["splits"]["train"] = train_split
        config["data"]["splits"]["val"] = train_split
        config["data"]["splits"]["test"] = test_split
        config["data"]["target_standardization"]["from_init_checkpoint"] = True
        config["train"]["epochs"] = 100
        config["train"]["output_dir"] = f"outputs/{name}"
        config["train"]["checkpoint_dir"] = f"outputs/{name}/checkpoints"
        config["train"]["init_checkpoint"] = f"outputs/{source_name}/checkpoints/best.pt"
        config["train"]["cache_frozen_image_features"] = True
        write_config(CONFIG_DIR / f"{name}.yaml", config)


if __name__ == "__main__":
    main()
