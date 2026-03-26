from __future__ import annotations

import math

import torch


UINT64_MASK = (1 << 64) - 1


def mix_seed(seed: int, *values: int) -> int:
    mixed = (seed + 0x9E3779B97F4A7C15) & UINT64_MASK
    for value in values:
        mixed ^= (value + 0x9E3779B97F4A7C15) & UINT64_MASK
        mixed = (mixed * 0xBF58476D1CE4E5B9) & UINT64_MASK
        mixed ^= mixed >> 30
        mixed = (mixed * 0x94D049BB133111EB) & UINT64_MASK
        mixed ^= mixed >> 31
    return mixed % (2**63 - 1)


def make_generator(seed: int, *values: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(mix_seed(seed, *values))
    return generator


def uniform_tensor(
    shape: tuple[int, ...],
    low: float,
    high: float,
    seed: int,
    *values: int,
) -> torch.Tensor:
    generator = make_generator(seed, *values)
    tensor = torch.rand(shape, generator=generator)
    return low + (high - low) * tensor


def sign_tensor(shape: tuple[int, ...], seed: int, *values: int) -> torch.Tensor:
    generator = make_generator(seed, *values)
    draws = torch.randint(0, 2, shape, generator=generator, dtype=torch.int64)
    return draws.to(torch.float32).mul_(2.0).sub_(1.0)


def fanin_scale(in_channels: int, kernel_size: int) -> float:
    return 1.0 / math.sqrt(in_channels * kernel_size * kernel_size)
