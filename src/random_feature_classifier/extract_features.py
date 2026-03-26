from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .cache import (
    choose_cache_dtype,
    default_cache_path,
    extract_split_features,
    save_feature_cache,
)
from .config import ModelConfig, TrainConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract and cache fixed top features for CIFAR-10.")
    parser.add_argument("--backbone-kind", choices=["strict_ones", "random_projection"], default="strict_ones")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width-multiplier", type=int, default=2)
    parser.add_argument("--tap-stages", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--global-feature-dim", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cache-dir", default="feature_cache")
    parser.add_argument("--cache-dtype", choices=["float16", "float32"], default="float16")
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
    cache_dtype = choose_cache_dtype(args.cache_dtype)

    for split in ("train", "test"):
        features, labels = extract_split_features(
            model_config=model_config,
            train_config=train_config,
            split=split,
            device=device,
            batch_size=args.batch_size,
            cache_dtype=cache_dtype,
        )
        output_path = default_cache_path(model_config, split, args.cache_dir)
        save_feature_cache(Path(output_path), features, labels, model_config, split)
        size_gb = features.numel() * features.element_size() / (1024**3)
        print(
            f"saved split={split} path={output_path} shape={tuple(features.shape)} "
            f"dtype={features.dtype} size_gb={size_gb:.2f}"
        )


if __name__ == "__main__":
    main()
