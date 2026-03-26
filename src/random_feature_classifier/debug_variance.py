from __future__ import annotations

import argparse

import torch
from torch.nn import functional as F

from .config import ModelConfig, TrainConfig
from .data import build_cifar10_datasets, cifar10_eval_transform, make_loader
from .layers import FixedFeatureProjector
from .backbones import normalize_pooled_stage_features
from .model import make_feature_extractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect variance through the fixed feature pipeline.")
    parser.add_argument("--backbone-kind", choices=["strict_ones", "random_projection"], default="strict_ones")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width-multiplier", type=int, default=2)
    parser.add_argument("--tap-stages", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--global-feature-dim", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-samples", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--device",
        default="cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu",
    )
    return parser.parse_args()


def flatten_features(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(tensor.shape[0], -1).to(torch.float32)


def summarize(name: str, tensor: torch.Tensor) -> None:
    matrix = flatten_features(tensor).to("cpu")
    column_std = matrix.std(dim=0)
    row_norm = matrix.norm(dim=1)
    print(
        f"{name}: shape={tuple(tensor.shape)} "
        f"mean={matrix.mean().item():.6f} std={matrix.std().item():.6f} "
        f"col_std_mean={column_std.mean().item():.6e} "
        f"col_std_min={column_std.min().item():.6e} "
        f"col_std_max={column_std.max().item():.6e} "
        f"row_norm_mean={row_norm.mean().item():.6f} "
        f"row_norm_std={row_norm.std().item():.6f}"
    )


def gather_inputs(train_config: TrainConfig, batch_size: int, num_samples: int) -> torch.Tensor:
    eval_transform = cifar10_eval_transform()
    train_dataset, _ = build_cifar10_datasets(
        train_config,
        train_transform=eval_transform,
        test_transform=eval_transform,
    )
    loader = make_loader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=train_config.num_workers)
    batches = []
    total = 0
    for images, _ in loader:
        batches.append(images)
        total += images.shape[0]
        if total >= num_samples:
            break
    return torch.cat(batches, dim=0)[:num_samples]


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(args.seed)

    model_config = ModelConfig(
        backbone_kind=args.backbone_kind,
        head_kind="linear",
        seed=args.seed,
        width_multiplier=args.width_multiplier,
        tap_stages=tuple(args.tap_stages),
        global_feature_dim=args.global_feature_dim,
    )
    train_config = TrainConfig(
        batch_size=args.batch_size,
        data_root=args.data_root,
        num_workers=args.num_workers,
    )

    device = torch.device(args.device)
    inputs = gather_inputs(train_config, args.batch_size, args.num_samples).to(device)
    feature_extractor = make_feature_extractor(model_config).to(device=device, memory_format=torch.channels_last)
    feature_extractor.eval()
    backbone = feature_extractor.backbone
    feature_map = feature_extractor.feature_map

    with torch.inference_mode():
        current = inputs.contiguous(memory_format=torch.channels_last)
        summarize("input", current)

        current = backbone.stem(current)
        current = backbone.stem_act(backbone.stem_norm(current))
        summarize("stem", current)

        pooled_parts = []
        for stage_index, stage in enumerate(backbone.stages, start=1):
            current = stage(current)
            summarize(f"stage{stage_index}", current)
            if stage_index in backbone.tap_stages:
                pooled = torch.flatten(F.adaptive_avg_pool2d(current, 1), 1)
                if backbone.normalize_tapped_pools:
                    pooled = normalize_pooled_stage_features(pooled)
                pooled_parts.append(pooled)

        pooled = torch.cat(pooled_parts, dim=1)
        summarize("pooled", pooled)

        if isinstance(feature_map, FixedFeatureProjector):
            projected = F.linear(pooled, feature_map.weight) / feature_map.scale
            summarize("projected", projected)
            activated = feature_map.activation(projected)
            summarize("activated", activated)
        else:
            summarize("activated", pooled)


if __name__ == "__main__":
    main()
