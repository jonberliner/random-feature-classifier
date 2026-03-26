from __future__ import annotations

from torch import nn

from .backbones import make_backbone
from .config import ModelConfig
from .heads import make_head
from .layers import FixedFeatureProjector


class RandomFeatureExtractor(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = make_backbone(config.backbone_kind, config)
        self.output_dim = self.backbone.output_dim

        if config.global_feature_dim > 0:
            self.feature_map = FixedFeatureProjector(
                in_dim=self.backbone.output_dim,
                out_dim=config.global_feature_dim,
                seed=config.seed,
                seed_offset=5000,
                sine_config=config.sine,
            )
            self.output_dim = config.global_feature_dim
        else:
            self.feature_map = nn.Identity()

    def forward_features(self, x):
        features = self.backbone(x)
        return self.feature_map(features)

    def forward(self, x):
        return self.forward_features(x)


class RandomFeatureClassifier(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.feature_extractor = RandomFeatureExtractor(config)
        self.backbone = self.feature_extractor.backbone
        self.feature_map = self.feature_extractor.feature_map
        self.head = make_head(config.head_kind, self.feature_extractor.output_dim, config)

    def forward_features(self, x):
        return self.feature_extractor(x)

    def forward(self, x):
        return self.head(self.forward_features(x))


def make_model(config: ModelConfig) -> RandomFeatureClassifier:
    return RandomFeatureClassifier(config)


def make_feature_extractor(config: ModelConfig) -> RandomFeatureExtractor:
    return RandomFeatureExtractor(config)
