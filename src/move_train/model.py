from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_input = self.norm1(x)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + attn_output
        return x + self.mlp(self.norm2(x))


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), dropout)

    def forward(self, image_tokens: torch.Tensor, physical_tokens: torch.Tensor) -> torch.Tensor:
        query = self.query_norm(image_tokens)
        context = self.context_norm(physical_tokens)
        attended, _ = self.attn(query, context, context, need_weights=False)
        image_tokens = image_tokens + attended
        return image_tokens + self.mlp(self.norm2(image_tokens))


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        image_size: tuple[int, int],
        patch_size: int,
        in_channels: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        height, width = image_size
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError(
                f"image_size {image_size} must be divisible by patch_size {patch_size}"
            )
        self.num_patches = (height // patch_size) * (width // patch_size)
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class CustomViTBackbone(nn.Module):
    def __init__(
        self,
        image_size: tuple[int, int],
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        image_depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.output_dim = embed_dim
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                SelfAttentionBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(image_depth)
            ]
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        batch_size = image.shape[0]
        image_tokens = self.patch_embed(image)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        image_tokens = torch.cat([cls_tokens, image_tokens], dim=1)
        image_tokens = self.dropout(image_tokens + self.position_embed)
        for block in self.blocks:
            image_tokens = block(image_tokens)
        return image_tokens


class TorchvisionBackbone(nn.Module):
    VIT_WEIGHTS_BY_NAME = {
        "vit_b_16": "ViT_B_16_Weights",
        "vit_b_32": "ViT_B_32_Weights",
        "vit_l_16": "ViT_L_16_Weights",
        "vit_l_32": "ViT_L_32_Weights",
    }
    CNN_WEIGHTS_BY_NAME = {
        "resnet50": "ResNet50_Weights",
        "efficientnet_b0": "EfficientNet_B0_Weights",
        "convnext_tiny": "ConvNeXt_Tiny_Weights",
    }

    def __init__(
        self,
        name: str,
        pretrained: bool = True,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        try:
            import torchvision.models as models
        except ImportError as exc:
            raise ImportError(
                "torchvision is required for pretrained image backbones. "
                "Install the project environment from environment.yml or install torchvision."
            ) from exc

        if not hasattr(models, name):
            choices = ", ".join(
                sorted(set(self.VIT_WEIGHTS_BY_NAME) | set(self.CNN_WEIGHTS_BY_NAME))
            )
            raise ValueError(f"Unsupported torchvision backbone {name!r}. Choices: {choices}")

        weights = None
        if pretrained:
            weights_name = self.VIT_WEIGHTS_BY_NAME.get(name, self.CNN_WEIGHTS_BY_NAME.get(name))
            if weights_name is None or not hasattr(models, weights_name):
                raise ValueError(f"No torchvision weights enum is configured for {name!r}")
            weights = getattr(models, weights_name).DEFAULT

        self.model = getattr(models, name)(weights=weights)
        self.is_vit = name in self.VIT_WEIGHTS_BY_NAME
        if self.is_vit:
            self.output_dim = int(self.model.hidden_dim)
        elif name.startswith("resnet"):
            self.output_dim = int(self.model.fc.in_features)
            self.model.fc = nn.Identity()
        elif name.startswith("efficientnet"):
            self.output_dim = int(self.model.classifier[-1].in_features)
            self.model.classifier = nn.Identity()
        elif name.startswith("convnext"):
            self.output_dim = int(self.model.classifier[-1].in_features)
            self.model.classifier[-1] = nn.Identity()
        else:
            raise ValueError(f"Unsupported torchvision backbone head layout for {name!r}")

        if not trainable:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if not self.is_vit:
            return self.model(image).unsqueeze(1)
        tokens = self.model._process_input(image)
        batch_size = tokens.shape[0]
        cls_token = self.model.class_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_token, tokens], dim=1)
        return self.model.encoder(tokens)


class TimmBackbone(nn.Module):
    def __init__(
        self,
        name: str,
        image_size: tuple[int, int],
        pretrained: bool = True,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "timm is required for timm image backbones. "
                "Install the project environment from environment.yml or install timm."
            ) from exc

        self.model = timm.create_model(
            name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        self.output_dim = self._infer_output_dim(image_size)

        if not trainable:
            for param in self.model.parameters():
                param.requires_grad = False

    def _forward_features(self, image: torch.Tensor) -> torch.Tensor:
        features = self.model(image)
        if features.ndim == 4:
            features = features.mean(dim=(-2, -1))
        elif features.ndim == 3:
            features = features.mean(dim=1)
        return features

    def _infer_output_dim(self, image_size: tuple[int, int]) -> int:
        was_training = self.model.training
        self.model.eval()
        height, width = image_size
        with torch.no_grad():
            features = self._forward_features(torch.zeros(1, 3, height, width))
        self.model.train(was_training)
        return int(features.shape[-1])

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self._forward_features(image)
        return features.unsqueeze(1)


@dataclass(frozen=True)
class ModelOutputShape:
    rows: int = 5
    cols: int = 9


class MultiModalTransformer(nn.Module):
    def __init__(
        self,
        image_size: tuple[int, int] | list[int] = (224, 224),
        image_backbone: str = "custom",
        pretrained_backbone: bool = False,
        trainable_backbone: bool = True,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 128,
        image_depth: int = 4,
        physical_depth: int = 1,
        fusion_depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        num_envs: int = 2,
        num_speeds: int = 8,
        num_hrr: int = 7,
        use_continuous_physics: bool = False,
        speed_scale: float = 7.0,
        hrr_scale: float = 6.0,
        output_rows: int = 5,
        output_cols: int = 9,
        input_mode: str = "multimodal",
        fusion_mode: str = "cross_attention",
    ) -> None:
        super().__init__()
        image_size = tuple(image_size)
        allowed_input_modes = {"multimodal", "image_only", "physics_only"}
        if input_mode not in allowed_input_modes:
            choices = ", ".join(sorted(allowed_input_modes))
            raise ValueError(f"Unsupported input_mode {input_mode!r}. Choices: {choices}")
        allowed_fusion_modes = {"cross_attention", "concat"}
        if fusion_mode not in allowed_fusion_modes:
            choices = ", ".join(sorted(allowed_fusion_modes))
            raise ValueError(f"Unsupported fusion_mode {fusion_mode!r}. Choices: {choices}")
        if input_mode != "multimodal" and fusion_mode != "cross_attention":
            raise ValueError("fusion_mode only applies when input_mode is 'multimodal'.")
        self.input_mode = input_mode
        self.fusion_mode = fusion_mode
        self.uses_image = input_mode in {"multimodal", "image_only"}
        self.uses_physics = input_mode in {"multimodal", "physics_only"}
        self.backbone_trainable = bool(trainable_backbone)
        self.use_continuous_physics = bool(use_continuous_physics)
        self.speed_scale = float(speed_scale)
        self.hrr_scale = float(hrr_scale)
        self.output_shape = ModelOutputShape(rows=output_rows, cols=output_cols)
        if self.uses_image and image_backbone == "custom":
            self.image_backbone = CustomViTBackbone(
                image_size=image_size,
                patch_size=patch_size,
                in_channels=in_channels,
                embed_dim=embed_dim,
                image_depth=image_depth,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
        elif self.uses_image and image_backbone.startswith("timm:"):
            self.image_backbone = TimmBackbone(
                name=image_backbone.removeprefix("timm:"),
                image_size=image_size,
                pretrained=pretrained_backbone,
                trainable=trainable_backbone,
            )
        elif self.uses_image:
            self.image_backbone = TorchvisionBackbone(
                name=image_backbone,
                pretrained=pretrained_backbone,
                trainable=trainable_backbone,
            )
        else:
            self.image_backbone = None
        if self.uses_image and not self.backbone_trainable:
            for param in self.image_backbone.parameters():
                param.requires_grad = False
        if self.uses_image:
            backbone_dim = int(self.image_backbone.output_dim)
            self.image_projection = (
                nn.Identity() if backbone_dim == embed_dim else nn.Linear(backbone_dim, embed_dim)
            )
        else:
            self.image_projection = None

        if self.uses_physics:
            self.env_embedding = nn.Embedding(num_envs, embed_dim)
            if self.use_continuous_physics:
                self.speed_projection = nn.Sequential(
                    nn.Linear(1, embed_dim),
                    nn.GELU(),
                    nn.Linear(embed_dim, embed_dim),
                )
                self.hrr_projection = nn.Sequential(
                    nn.Linear(1, embed_dim),
                    nn.GELU(),
                    nn.Linear(embed_dim, embed_dim),
                )
            else:
                self.speed_embedding = nn.Embedding(num_speeds, embed_dim)
                self.hrr_embedding = nn.Embedding(num_hrr, embed_dim)
            self.position_projection = nn.Sequential(
                nn.Linear(1, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )
            self.physical_type_embedding = nn.Parameter(torch.zeros(1, 4, embed_dim))
            self.physical_fusion = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, embed_dim),
            )
            self.physical_blocks = nn.ModuleList(
                [
                    SelfAttentionBlock(embed_dim, num_heads, mlp_ratio, dropout)
                    for _ in range(physical_depth)
                ]
            )
        else:
            self.env_embedding = None
            self.physical_blocks = nn.ModuleList()
        if self.input_mode == "multimodal" and self.fusion_mode == "cross_attention":
            self.fusion_blocks = nn.ModuleList(
                [
                    CrossAttentionBlock(embed_dim, num_heads, mlp_ratio, dropout)
                    for _ in range(fusion_depth)
                ]
            )
        else:
            self.fusion_blocks = nn.ModuleList()
        self.norm = nn.LayerNorm(embed_dim)
        if self.input_mode == "multimodal" and self.fusion_mode == "concat":
            self.concat_fusion = nn.Sequential(
                nn.LayerNorm(embed_dim * 2),
                nn.Linear(embed_dim * 2, embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, embed_dim),
            )
        else:
            self.concat_fusion = None
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, output_rows * output_cols),
        )
        self._init_new_weights(include_backbone=image_backbone == "custom")

    def _init_module_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _init_new_weights(self, include_backbone: bool) -> None:
        if include_backbone and isinstance(self.image_backbone, CustomViTBackbone):
            nn.init.trunc_normal_(self.image_backbone.cls_token, std=0.02)
            nn.init.trunc_normal_(self.image_backbone.position_embed, std=0.02)
            self.image_backbone.apply(self._init_module_weights)
        if isinstance(self.image_projection, nn.Linear):
            self.image_projection.apply(self._init_module_weights)
        if self.uses_physics:
            nn.init.trunc_normal_(self.physical_type_embedding, std=0.02)
            self.env_embedding.reset_parameters()
            if self.use_continuous_physics:
                self.speed_projection.apply(self._init_module_weights)
                self.hrr_projection.apply(self._init_module_weights)
            else:
                self.speed_embedding.reset_parameters()
                self.hrr_embedding.reset_parameters()
            self.position_projection.apply(self._init_module_weights)
            self.physical_fusion.apply(self._init_module_weights)
            self.physical_blocks.apply(self._init_module_weights)
        self.fusion_blocks.apply(self._init_module_weights)
        if self.concat_fusion is not None:
            self.concat_fusion.apply(self._init_module_weights)
        self.head.apply(self._init_module_weights)

    def train(self, mode: bool = True) -> "MultiModalTransformer":
        super().train(mode)
        if self.uses_image and not self.backbone_trainable:
            self.image_backbone.eval()
        return self

    def encode_physical(
        self,
        env: torch.Tensor,
        speed: torch.Tensor,
        hrr: torch.Tensor,
        position: torch.Tensor,
        speed_value: torch.Tensor | None = None,
        hrr_value: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.uses_physics:
            raise RuntimeError("encode_physical was called for an image-only model.")
        position = position.float().view(-1, 1)
        if self.use_continuous_physics:
            if speed_value is None:
                speed_value = speed.float()
            if hrr_value is None:
                hrr_value = hrr.float()
            speed_token = self.speed_projection(speed_value.float().view(-1, 1) / self.speed_scale)
            hrr_token = self.hrr_projection(hrr_value.float().view(-1, 1) / self.hrr_scale)
        else:
            speed_token = self.speed_embedding(speed.long())
            hrr_token = self.hrr_embedding(hrr.long())
        tokens = torch.stack(
            [
                self.env_embedding(env.long()),
                speed_token,
                hrr_token,
                self.position_projection(position),
            ],
            dim=1,
        )
        tokens = tokens + self.physical_type_embedding
        tokens = tokens + self.physical_fusion(tokens)
        for block in self.physical_blocks:
            tokens = block(tokens)
        return tokens

    def forward(
        self,
        image: torch.Tensor,
        env: torch.Tensor,
        speed: torch.Tensor,
        hrr: torch.Tensor,
        position: torch.Tensor,
        speed_value: torch.Tensor | None = None,
        hrr_value: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.uses_image:
            if self.backbone_trainable:
                image_features = self.image_backbone(image)
            else:
                with torch.no_grad():
                    image_features = self.image_backbone(image)
        else:
            image_features = None
        return self.forward_from_image_features(
            image_features=image_features,
            env=env,
            speed=speed,
            hrr=hrr,
            position=position,
            speed_value=speed_value,
            hrr_value=hrr_value,
        )

    def forward_from_image_features(
        self,
        image_features: torch.Tensor | None,
        env: torch.Tensor,
        speed: torch.Tensor,
        hrr: torch.Tensor,
        position: torch.Tensor,
        speed_value: torch.Tensor | None = None,
        hrr_value: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.uses_image:
            if image_features is None:
                raise RuntimeError("image_features are required for this model.")
            batch_size = image_features.shape[0]
            image_tokens = self.image_projection(image_features)
        else:
            batch_size = env.shape[0]
            image_tokens = None

        physical_tokens = None
        if self.uses_physics:
            physical_tokens = self.encode_physical(
                env, speed, hrr, position, speed_value, hrr_value
            )

        if self.input_mode == "image_only":
            pooled = self.norm(image_tokens[:, 0])
        elif self.input_mode == "physics_only":
            pooled = self.norm(physical_tokens.mean(dim=1))
        elif self.fusion_mode == "concat":
            image_pooled = image_tokens[:, 0]
            physical_pooled = physical_tokens.mean(dim=1)
            pooled = self.norm(self.concat_fusion(torch.cat([image_pooled, physical_pooled], dim=1)))
        else:
            for block in self.fusion_blocks:
                image_tokens = block(image_tokens, physical_tokens)
            pooled = self.norm(image_tokens[:, 0])
        output = self.head(pooled)
        return output.view(batch_size, self.output_shape.rows, self.output_shape.cols)
