from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig
from .layers import FixedChannelNorm, FixedConv2d, FixedSineActivation


def normalize_pooled_stage_features(features: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(features, (features.shape[1],))


class FixedBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        config: ModelConfig,
        seed_offset: int,
    ) -> None:
        super().__init__()
        self.conv1 = FixedConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            seed=config.seed,
            seed_offset=seed_offset,
            mode=config.backbone_kind,
            scale_by_fanin=config.scale_by_fanin,
        )
        self.act1 = FixedSineActivation(
            num_features=out_channels,
            seed=config.seed,
            seed_offset=seed_offset + 1,
            sine_config=config.sine,
            spatial=True,
        )
        self.norm1 = FixedChannelNorm(out_channels)
        self.conv2 = FixedConv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            seed=config.seed,
            seed_offset=seed_offset + 2,
            mode=config.backbone_kind,
            scale_by_fanin=config.scale_by_fanin,
        )
        self.act2 = FixedSineActivation(
            num_features=out_channels,
            seed=config.seed,
            seed_offset=seed_offset + 3,
            sine_config=config.sine,
            spatial=True,
        )
        self.norm2 = FixedChannelNorm(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = FixedConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                seed=config.seed,
                seed_offset=seed_offset + 4,
                mode=config.backbone_kind,
                scale_by_fanin=config.scale_by_fanin,
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.act1(self.norm1(out))
        out = self.conv2(out)
        out = self.act2(self.norm2(out))
        return residual + out


class FixedResNetBackbone(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        stage_channels = config.scaled_stage_channels()
        self.tap_stages = tuple(config.tap_stages)
        self.normalize_tapped_pools = config.normalize_tapped_pools
        self.output_dim = config.backbone_output_dim()
        self.stem = FixedConv2d(
            in_channels=3,
            out_channels=stage_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            seed=config.seed,
            seed_offset=0,
            mode=config.backbone_kind,
            scale_by_fanin=config.scale_by_fanin,
        )
        self.stem_act = FixedSineActivation(
            num_features=stage_channels[0],
            seed=config.seed,
            seed_offset=1,
            sine_config=config.sine,
            spatial=True,
        )
        self.stem_norm = FixedChannelNorm(stage_channels[0])
        self.stages = nn.ModuleList()

        in_channels = stage_channels[0]
        seed_cursor = 10
        for stage_index, (out_channels, block_count) in enumerate(
            zip(stage_channels, config.blocks_per_stage, strict=True)
        ):
            blocks = []
            for block_index in range(block_count):
                stride = 2 if stage_index > 0 and block_index == 0 else 1
                blocks.append(
                    FixedBasicBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=stride,
                        config=config,
                        seed_offset=seed_cursor,
                    )
                )
                seed_cursor += 10
                in_channels = out_channels
            self.stages.append(nn.Sequential(*blocks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        out = self.stem_act(self.stem_norm(out))
        pooled_features = []
        for stage_index, stage in enumerate(self.stages, start=1):
            out = stage(out)
            if stage_index in self.tap_stages:
                pooled = torch.flatten(F.adaptive_avg_pool2d(out, 1), 1)
                if self.normalize_tapped_pools:
                    pooled = normalize_pooled_stage_features(pooled)
                pooled_features.append(pooled)

        if not pooled_features:
            raise ValueError("ModelConfig.tap_stages must contain at least one stage index.")
        return torch.cat(pooled_features, dim=1)


def make_backbone(kind: str, config: ModelConfig) -> nn.Module:
    if kind not in {"strict_ones", "random_projection"}:
        raise ValueError(f"Unknown backbone kind: {kind}")
    return FixedResNetBackbone(replace(config, backbone_kind=kind))
