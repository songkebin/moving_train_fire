from __future__ import annotations

import torch

from move_train.model import MultiModalTransformer


def test_model_forward_shape_and_backward() -> None:
    model = MultiModalTransformer(
        image_size=(32, 32),
        image_backbone="custom",
        pretrained_backbone=False,
        patch_size=8,
        embed_dim=32,
        image_depth=1,
        physical_depth=1,
        fusion_depth=1,
        num_heads=4,
        dropout=0.0,
    )
    prediction = model(
        image=torch.randn(2, 3, 32, 32),
        env=torch.tensor([0, 1]),
        speed=torch.tensor([1, 7]),
        hrr=torch.tensor([1, 6]),
        position=torch.tensor([0.1, 0.9]),
    )

    assert prediction.shape == (2, 5, 9)
    loss = prediction.square().mean()
    loss.backward()
    assert all(param.grad is not None for param in model.parameters() if param.requires_grad)


def test_model_accepts_continuous_physical_values() -> None:
    model = MultiModalTransformer(
        image_size=(32, 32),
        image_backbone="custom",
        pretrained_backbone=False,
        trainable_backbone=False,
        patch_size=8,
        embed_dim=32,
        image_depth=1,
        physical_depth=1,
        fusion_depth=1,
        num_heads=4,
        dropout=0.0,
        use_continuous_physics=True,
        speed_scale=7.0,
        hrr_scale=6.0,
    )
    prediction = model(
        image=torch.randn(2, 3, 32, 32),
        env=torch.tensor([0, 1]),
        speed=torch.tensor([1, 7]),
        hrr=torch.tensor([0, 5]),
        speed_value=torch.tensor([1.0, 7.0]),
        hrr_value=torch.tensor([1.0, 6.0]),
        position=torch.tensor([0.1, 0.9]),
    )

    assert prediction.shape == (2, 5, 9)
    loss = prediction.square().mean()
    loss.backward()
    assert all(param.grad is not None for param in model.parameters() if param.requires_grad)
    assert all(param.grad is None for param in model.image_backbone.parameters())


def test_model_supports_input_ablation_modes() -> None:
    common_kwargs = {
        "image_size": (32, 32),
        "image_backbone": "custom",
        "pretrained_backbone": False,
        "patch_size": 8,
        "embed_dim": 32,
        "image_depth": 1,
        "physical_depth": 1,
        "fusion_depth": 1,
        "num_heads": 4,
        "dropout": 0.0,
        "use_continuous_physics": True,
    }
    batch = {
        "image": torch.randn(2, 3, 32, 32),
        "env": torch.tensor([0, 1]),
        "speed": torch.tensor([1, 7]),
        "hrr": torch.tensor([0, 5]),
        "speed_value": torch.tensor([1.0, 7.0]),
        "hrr_value": torch.tensor([1.0, 6.0]),
        "position": torch.tensor([0.1, 0.9]),
    }

    for kwargs in (
        {"input_mode": "image_only"},
        {"input_mode": "physics_only"},
        {"input_mode": "multimodal", "fusion_mode": "concat"},
    ):
        model = MultiModalTransformer(**common_kwargs, **kwargs)
        prediction = model(**batch)

        assert prediction.shape == (2, 5, 9)
        loss = prediction.square().mean()
        loss.backward()
        assert any(param.grad is not None for param in model.parameters() if param.requires_grad)
