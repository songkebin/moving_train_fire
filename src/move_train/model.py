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


class TorchvisionViTBackbone(nn.Module):
    WEIGHTS_BY_NAME = {
        "vit_b_16": "ViT_B_16_Weights",
        "vit_b_32": "ViT_B_32_Weights",
        "vit_l_16": "ViT_L_16_Weights",
        "vit_l_32": "ViT_L_32_Weights",
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
                "torchvision is required for pretrained ViT backbones. "
                "Install the project environment from environment.yml or install torchvision."
            ) from exc

        if not hasattr(models, name):
            choices = ", ".join(sorted(self.WEIGHTS_BY_NAME))
            raise ValueError(f"Unsupported torchvision ViT backbone {name!r}. Choices: {choices}")

        weights = None
        if pretrained:
            weights_name = self.WEIGHTS_BY_NAME.get(name)
            if weights_name is None or not hasattr(models, weights_name):
                raise ValueError(f"No torchvision weights enum is configured for {name!r}")
            weights = getattr(models, weights_name).DEFAULT

        self.model = getattr(models, name)(weights=weights)
        self.output_dim = int(self.model.hidden_dim)

        if not trainable:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        tokens = self.model._process_input(image)
        batch_size = tokens.shape[0]
        cls_token = self.model.class_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_token, tokens], dim=1)
        return self.model.encoder(tokens)


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
        output_rows: int = 5,
        output_cols: int = 9,
    ) -> None:
        super().__init__()
        image_size = tuple(image_size)
        self.output_shape = ModelOutputShape(rows=output_rows, cols=output_cols)
        if image_backbone == "custom":
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
        else:
            self.image_backbone = TorchvisionViTBackbone(
                name=image_backbone,
                pretrained=pretrained_backbone,
                trainable=trainable_backbone,
            )
        backbone_dim = int(self.image_backbone.output_dim)
        self.image_projection = (
            nn.Identity() if backbone_dim == embed_dim else nn.Linear(backbone_dim, embed_dim)
        )

        self.env_embedding = nn.Embedding(num_envs, embed_dim)
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
        self.fusion_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(fusion_depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
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
        nn.init.trunc_normal_(self.physical_type_embedding, std=0.02)
        self.env_embedding.reset_parameters()
        self.speed_embedding.reset_parameters()
        self.hrr_embedding.reset_parameters()
        self.position_projection.apply(self._init_module_weights)
        self.physical_fusion.apply(self._init_module_weights)
        self.physical_blocks.apply(self._init_module_weights)
        self.fusion_blocks.apply(self._init_module_weights)
        self.head.apply(self._init_module_weights)

    def encode_physical(
        self,
        env: torch.Tensor,
        speed: torch.Tensor,
        hrr: torch.Tensor,
        position: torch.Tensor,
    ) -> torch.Tensor:
        position = position.float().view(-1, 1)
        tokens = torch.stack(
            [
                self.env_embedding(env.long()),
                self.speed_embedding(speed.long()),
                self.hrr_embedding(hrr.long()),
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
    ) -> torch.Tensor:
        batch_size = image.shape[0]
        image_tokens = self.image_projection(self.image_backbone(image))

        physical_tokens = self.encode_physical(env, speed, hrr, position)
        for block in self.fusion_blocks:
            image_tokens = block(image_tokens, physical_tokens)

        pooled = self.norm(image_tokens[:, 0])
        output = self.head(pooled)
        return output.view(batch_size, self.output_shape.rows, self.output_shape.cols)
