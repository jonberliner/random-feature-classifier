from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import ModelConfig, TrainConfig
from .data import build_cifar10_datasets, cifar10_eval_transform, make_loader
from .model import make_feature_extractor


def default_cache_path(config: ModelConfig, split: str, directory: str) -> Path:
    tap_suffix = "-".join(str(stage_index) for stage_index in config.tap_stages)
    filename = (
        f"{split}_features_"
        f"{config.backbone_kind}_"
        f"t{tap_suffix}_"
        f"w{config.width_multiplier}_"
        f"g{config.global_feature_dim}_"
        f"s{config.seed}.pt"
    )
    return Path(directory) / filename


def default_topk_cache_path(
    directory: str,
    split: str,
    backbone_kind: str,
    layer_specs: list[object],
    pairwise_per_pair_keeps: list[int],
    seed: int,
    standardized: bool,
) -> Path:
    candidates = "-".join(str(spec.num_candidates) for spec in layer_specs)
    keeps = "-".join(str(spec.keep_k) for spec in layer_specs)
    strides = "-".join(str(spec.stride) for spec in layer_specs)
    pairwise = "-".join(str(value) for value in pairwise_per_pair_keeps)
    standardized_suffix = "std" if standardized else "raw"
    filename = (
        f"{split}_topk_refine_{backbone_kind}_"
        f"c{candidates}_g{keeps}_p{pairwise}_s{strides}_"
        f"{standardized_suffix}_seed{seed}.pt"
    )
    return Path(directory) / filename


def choose_cache_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported cache dtype: {name}")


def make_feature_loader_for_split(
    train_config: TrainConfig,
    split: str,
    batch_size: int,
) -> DataLoader:
    eval_transform = cifar10_eval_transform()
    train_dataset, test_dataset = build_cifar10_datasets(
        train_config,
        train_transform=eval_transform,
        test_transform=eval_transform,
    )
    dataset = train_dataset if split == "train" else test_dataset
    return make_loader(dataset, batch_size=batch_size, shuffle=False, num_workers=train_config.num_workers)


def extract_split_features(
    model_config: ModelConfig,
    train_config: TrainConfig,
    split: str,
    device: torch.device,
    batch_size: int,
    cache_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    loader = make_feature_loader_for_split(train_config, split, batch_size=batch_size)
    feature_extractor = make_feature_extractor(model_config).to(device=device, memory_format=torch.channels_last)
    feature_extractor.eval()

    features_cpu: torch.Tensor | None = None
    labels_cpu = torch.empty(len(loader.dataset), dtype=torch.int64)
    cursor = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            batch_features = feature_extractor(images).to("cpu", dtype=cache_dtype)
            batch_size_actual = batch_features.shape[0]
            if features_cpu is None:
                features_cpu = torch.empty(
                    (len(loader.dataset), batch_features.shape[1]),
                    dtype=cache_dtype,
                )
            features_cpu[cursor : cursor + batch_size_actual].copy_(batch_features)
            labels_cpu[cursor : cursor + batch_size_actual].copy_(labels.to(torch.int64))
            cursor += batch_size_actual

    if features_cpu is None:
        raise RuntimeError("No features were extracted.")
    return features_cpu, labels_cpu


def save_feature_cache(
    output_path: Path,
    features: torch.Tensor,
    labels: torch.Tensor,
    model_config: ModelConfig,
    split: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "features": features,
            "labels": labels,
            "split": split,
            "backbone_kind": model_config.backbone_kind,
            "tap_stages": model_config.tap_stages,
            "width_multiplier": model_config.width_multiplier,
            "global_feature_dim": model_config.global_feature_dim,
            "seed": model_config.seed,
        },
        output_path,
    )


def load_feature_cache(path: str) -> tuple[torch.Tensor, torch.Tensor, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return payload["features"], payload["labels"], payload


def load_matrix_feature_cache(path: str) -> tuple[torch.Tensor, torch.Tensor, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if "features" in payload:
        return payload["features"], payload["labels"], payload
    if "global_features" in payload:
        return payload["global_features"], payload["labels"], payload
    raise ValueError(f"Unsupported feature cache format: {path}")


def save_topk_feature_cache(
    output_path: Path,
    global_features: torch.Tensor,
    pairwise_features: torch.Tensor,
    labels: torch.Tensor,
    metadata: dict,
    split: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "global_features": global_features,
            "pairwise_features": pairwise_features,
            "labels": labels,
            "split": split,
            **metadata,
        },
        output_path,
    )


def load_topk_feature_cache(path: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return payload["global_features"], payload["pairwise_features"], payload["labels"], payload
