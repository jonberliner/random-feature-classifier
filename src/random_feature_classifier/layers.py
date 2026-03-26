from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import SineConfig
from .prng import sign_tensor, uniform_tensor


class FixedSineActivation(nn.Module):
    def __init__(
        self,
        num_features: int,
        seed: int,
        seed_offset: int,
        sine_config: SineConfig,
        spatial: bool,
    ) -> None:
        super().__init__()
        view_shape = (1, num_features, 1, 1) if spatial else (1, num_features)
        a = uniform_tensor(
            (num_features,),
            sine_config.a_min,
            sine_config.a_max,
            seed,
            seed_offset,
            0,
        ).view(view_shape)
        b = uniform_tensor(
            (num_features,),
            sine_config.b_min,
            sine_config.b_max,
            seed,
            seed_offset,
            1,
        ).view(view_shape)
        c = uniform_tensor(
            (num_features,),
            sine_config.c_min,
            sine_config.c_max,
            seed,
            seed_offset,
            2,
        ).view(view_shape)
        self.register_buffer("a", a, persistent=False)
        self.register_buffer("b", b, persistent=False)
        self.register_buffer("c", c, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c * torch.sin(self.a * x + self.b)


def choose_group_count(num_features: int) -> int:
    for group_count in (32, 16, 8, 4, 2):
        if num_features % group_count == 0:
            return group_count
    return 1


class FixedChannelNorm(nn.Module):
    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(
            num_groups=choose_group_count(num_features),
            num_channels=num_features,
            affine=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class FixedConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        seed: int,
        seed_offset: int,
        mode: str,
        scale_by_fanin: bool,
    ) -> None:
        super().__init__()
        shape = (out_channels, in_channels, kernel_size, kernel_size)
        if mode == "strict_ones":
            weight = torch.ones(shape, dtype=torch.float32)
        elif mode == "random_projection":
            weight = sign_tensor(shape, seed, seed_offset, 0)
        else:
            raise ValueError(f"Unknown fixed conv mode: {mode}")

        self.register_buffer("weight", weight, persistent=False)
        self.stride = stride
        self.padding = padding
        self.scale = float(in_channels * kernel_size * kernel_size) if scale_by_fanin else 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, stride=self.stride, padding=self.padding) / self.scale


class FixedFeatureProjector(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        seed: int,
        seed_offset: int,
        sine_config: SineConfig,
    ) -> None:
        super().__init__()
        weight = sign_tensor((out_dim, in_dim), seed, seed_offset, 0)
        self.register_buffer("weight", weight, persistent=False)
        self.activation = FixedSineActivation(
            num_features=out_dim,
            seed=seed,
            seed_offset=seed_offset + 1,
            sine_config=sine_config,
            spatial=False,
        )
        self.scale = math.sqrt(float(max(1, in_dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = F.linear(x, self.weight) / self.scale
        return self.activation(projected)
