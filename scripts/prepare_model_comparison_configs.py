from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


CONFIG_DIR = Path("configs")

BASE_SOURCE = CONFIG_DIR / "exp_tunnel_random622_to_open_full_100ep.yaml"
BASE_RANDOM = CONFIG_DIR / "exp_open_random20_finetune_from_tunnel_random622_100ep.yaml"
BASE_CONDITION = CONFIG_DIR / "exp_open_condition20_finetune_from_tunnel_random622_100ep.yaml"

VARIANTS = {
    "physics_only": {
        "label": "Physics-only MLP",
        "model": {
            "input_mode": "physics_only",
        },
        "cache_frozen_image_features": False,
    },
    "image_only_vit_b16": {
        "label": "Image-only ViT-B/16",
        "model": {
            "input_mode": "image_only",
            "image_backbone": "vit_b_16",
            "embed_dim": 768,
            "num_heads": 8,
        },
        "cache_frozen_image_features": True,
    },
    "image_only_resnet50": {
        "label": "Image-only ResNet50",
        "model": {
            "input_mode": "image_only",
            "image_backbone": "resnet50",
            "embed_dim": 2048,
            "num_heads": 8,
        },
        "cache_frozen_image_features": True,
    },
    "image_only_efficientnet_b0": {
        "label": "Image-only EfficientNet-B0",
        "model": {
            "input_mode": "image_only",
            "image_backbone": "efficientnet_b0",
            "embed_dim": 1280,
            "num_heads": 8,
        },
        "cache_frozen_image_features": True,
    },
    "image_only_convnext_tiny": {
        "label": "Image-only ConvNeXt-Tiny",
        "model": {
            "input_mode": "image_only",
            "image_backbone": "convnext_tiny",
            "embed_dim": 768,
            "num_heads": 8,
        },
        "cache_frozen_image_features": True,
    },
    "image_only_swin_tiny": {
        "label": "Image-only Swin-Tiny",
        "model": {
            "input_mode": "image_only",
            "image_backbone": "timm:swin_tiny_patch4_window7_224.ms_in1k",
            "embed_dim": 768,
            "num_heads": 8,
        },
        "cache_frozen_image_features": True,
    },
    "image_only_pvt_v2_b2": {
        "label": "Image-only PVTv2-B2",
        "model": {
            "input_mode": "image_only",
            "image_backbone": "timm:pvt_v2_b2.in1k",
            "embed_dim": 512,
            "num_heads": 8,
        },
        "cache_frozen_image_features": True,
    },
    "image_only_maxvit_tiny": {
        "label": "Image-only MaxViT-Tiny",
        "model": {
            "input_mode": "image_only",
            "image_backbone": "timm:maxvit_tiny_tf_224.in1k",
            "embed_dim": 512,
            "num_heads": 8,
        },
        "cache_frozen_image_features": True,
    },
    "image_only_mambaout_tiny": {
        "label": "Image-only MambaOut-Tiny",
        "model": {
            "input_mode": "image_only",
            "image_backbone": "timm:mambaout_tiny.in1k",
            "embed_dim": 576,
            "num_heads": 8,
        },
        "cache_frozen_image_features": True,
    },
    "concat_vit_b16": {
        "label": "ViT-B/16 concat fusion",
        "model": {
            "input_mode": "multimodal",
            "fusion_mode": "concat",
        },
        "cache_frozen_image_features": True,
    },
}

EXPERIMENTS = {
    "open_full": {
        "base": BASE_SOURCE,
        "suffix": "tunnel_random622_to_open_full_100ep",
        "init_from_source": False,
    },
    "open_random20": {
        "base": BASE_RANDOM,
        "suffix": "open_random20_finetune_from_tunnel_random622_100ep",
        "init_from_source": True,
    },
    "open_condition20": {
        "base": BASE_CONDITION,
        "suffix": "open_condition20_finetune_from_tunnel_random622_100ep",
        "init_from_source": True,
    },
}


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def build_config(base: dict, variant_name: str, variant: dict, experiment: dict) -> dict:
    config = deepcopy(base)
    config["model"].update(variant["model"])

    output_dir = Path("outputs") / f"exp_cmp_{variant_name}_{experiment['suffix']}"
    config["train"]["output_dir"] = str(output_dir)
    config["train"]["checkpoint_dir"] = str(output_dir / "checkpoints")
    config["train"]["cache_frozen_image_features"] = bool(
        variant["cache_frozen_image_features"]
    )

    if experiment["init_from_source"]:
        source_dir = Path("outputs") / f"exp_cmp_{variant_name}_{EXPERIMENTS['open_full']['suffix']}"
        config["train"]["init_checkpoint"] = str(source_dir / "checkpoints" / "best.pt")
    else:
        config["train"].pop("init_checkpoint", None)

    return config


def main() -> None:
    for variant_name, variant in VARIANTS.items():
        for experiment_name, experiment in EXPERIMENTS.items():
            base = load_yaml(experiment["base"])
            config = build_config(base, variant_name, variant, experiment)
            path = CONFIG_DIR / f"exp_cmp_{variant_name}_{experiment['suffix']}.yaml"
            write_yaml(path, config)
            print(f"Wrote {path} ({variant['label']}, {experiment_name})")


if __name__ == "__main__":
    main()
