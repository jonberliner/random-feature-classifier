from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig
from .prng import uniform_tensor


class LinearReadoutHead(nn.Module):
    def __init__(self, in_dim: int, config: ModelConfig) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, config.num_classes)
        scale = 1.0 / math.sqrt(float(max(1, in_dim)))
        with torch.no_grad():
            self.linear.weight.copy_(
                uniform_tensor(
                    self.linear.weight.shape,
                    -scale,
                    scale,
                    config.seed,
                    1000,
                    0,
                )
            )
            self.linear.bias.copy_(
                uniform_tensor(
                    self.linear.bias.shape,
                    -scale,
                    scale,
                    config.seed,
                    1000,
                    1,
                )
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)


class ScalarClassHead(nn.Module):
    def __init__(self, in_dim: int, config: ModelConfig) -> None:
        super().__init__()
        self.raw_class_codes = nn.Parameter(torch.zeros(config.num_classes))
        alpha = uniform_tensor(
            (config.num_classes, in_dim),
            config.sine.a_min,
            config.sine.a_max,
            config.seed,
            2000,
            0,
        )
        beta = uniform_tensor(
            (config.num_classes, in_dim),
            config.sine.b_min,
            config.sine.b_max,
            config.seed,
            2000,
            1,
        )
        gamma = uniform_tensor(
            (config.num_classes, in_dim),
            config.sine.c_min,
            config.sine.c_max,
            config.seed,
            2000,
            2,
        )
        self.register_buffer("alpha", alpha, persistent=False)
        self.register_buffer("beta", beta, persistent=False)
        self.register_buffer("gamma", gamma, persistent=False)
        self.scale = math.sqrt(float(max(1, in_dim)))

    def prototypes(self) -> torch.Tensor:
        class_codes = torch.tanh(self.raw_class_codes).unsqueeze(1)
        return self.gamma * torch.sin(self.alpha * class_codes + self.beta)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.linear(features, self.prototypes()) / self.scale


def make_head(kind: str, in_dim: int, config: ModelConfig) -> nn.Module:
    if kind == "linear":
        return LinearReadoutHead(in_dim, config)
    if kind == "scalar":
        return ScalarClassHead(in_dim, config)
    raise ValueError(f"Unknown head kind: {kind}")
