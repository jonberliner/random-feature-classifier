from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SineConfig:
    a_min: float = 0.25
    a_max: float = 1.25
    b_min: float = -math.pi
    b_max: float = math.pi
    c_min: float = 0.5
    c_max: float = 1.5


@dataclass(frozen=True)
class ModelConfig:
    backbone_kind: str = "strict_ones"
    head_kind: str = "linear"
    seed: int = 0
    width_multiplier: int = 2
    stage_channels: tuple[int, int, int, int] = (64, 128, 256, 512)
    blocks_per_stage: tuple[int, int, int, int] = (2, 2, 2, 2)
    num_classes: int = 10
    tap_stages: tuple[int, ...] = (1, 2, 3, 4)
    normalize_tapped_pools: bool = True
    global_feature_dim: int = 4096
    scale_by_fanin: bool = False
    sine: SineConfig = field(default_factory=SineConfig)

    def scaled_stage_channels(self) -> tuple[int, int, int, int]:
        return tuple(channel * self.width_multiplier for channel in self.stage_channels)

    def stem_channels(self) -> int:
        return self.scaled_stage_channels()[0]

    def tapped_stage_channels(self) -> tuple[int, ...]:
        stage_channels = self.scaled_stage_channels()
        return tuple(stage_channels[index - 1] for index in self.tap_stages)

    def backbone_output_dim(self) -> int:
        return sum(self.tapped_stage_channels())

    def head_input_dim(self) -> int:
        if self.global_feature_dim > 0:
            return self.global_feature_dim
        return self.backbone_output_dim()


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 128
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    data_root: str = "data"
    num_workers: int = 2
    warmup_epochs: int = 1
    label_smoothing: float = 0.0
