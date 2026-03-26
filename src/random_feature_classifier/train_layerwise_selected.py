from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from .config import ModelConfig, SineConfig, TrainConfig
from .data import build_cifar10_datasets, build_feature_loader, cifar10_eval_transform, make_loader
from .heads import make_head
from .layerwise_selection import (
    LayerSelection,
    LayerSpec,
    LayerwiseRandomFeatureStack,
    cache_loader_tensors,
    expand_feature_family_masks,
    save_layerwise_selection,
)
from .train_head import make_scheduler, run_epoch, standardize_train_test


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def pad_optional_int_list(values: list[int], target_length: int) -> list[int]:
    if not values:
        return [0] * target_length
    if len(values) != target_length:
        raise ValueError("Optional layer-wise integer list must match the number of layers.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Greedy layerwise random-channel selection on CIFAR-10.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backbone-kind", choices=["strict_ones", "random_projection"], default="random_projection")
    parser.add_argument("--head-kind", choices=["linear", "scalar"], default="linear")
    parser.add_argument("--layer-candidates", type=parse_int_list, default=parse_int_list("256,256,256"))
    parser.add_argument("--layer-keeps", type=parse_int_list, default=parse_int_list("32,32,32"))
    parser.add_argument("--layer-strides", type=parse_int_list, default=parse_int_list("1,2,2"))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-abs-correlation", type=float, default=0.95)
    parser.add_argument(
        "--selection-strategy",
        choices=["multiclass", "hybrid_pairwise", "global_plus_pairwise_per_pair", "multiclass_pairwise_ovr_shortlist"],
        default="multiclass",
    )
    parser.add_argument("--pairwise-alpha", type=float, default=0.5)
    parser.add_argument("--multiclass-keep-fraction", type=float, default=0.5)
    parser.add_argument("--pairwise-keep-fraction", type=float, default=0.25)
    parser.add_argument("--ovr-keep-fraction", type=float, default=0.25)
    parser.add_argument("--selection-shortlist-multiplier", type=float, default=2.0)
    parser.add_argument("--pairwise-extra-keeps", type=parse_int_list, default=parse_int_list(""))
    parser.add_argument("--standardize-features", action="store_true")
    parser.add_argument("--selection-path", type=str)
    parser.add_argument("--save-feature-dir", type=str)
    parser.add_argument(
        "--device",
        default="cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu",
    )
    return parser.parse_args()


def build_layer_specs(candidates: list[int], keeps: list[int], strides: list[int]) -> list[LayerSpec]:
    if not (len(candidates) == len(keeps) == len(strides)):
        raise ValueError("Layer candidate, keep, and stride lists must have the same length.")
    return [
        LayerSpec(num_candidates=num_candidates, keep_k=keep_k, stride=stride)
        for num_candidates, keep_k, stride in zip(candidates, keeps, strides, strict=True)
    ]


def build_eval_loaders(train_config: TrainConfig):
    eval_transform = cifar10_eval_transform()
    train_dataset, test_dataset = build_cifar10_datasets(
        train_config,
        train_transform=eval_transform,
        test_transform=eval_transform,
    )
    train_loader = make_loader(train_dataset, batch_size=train_config.batch_size, shuffle=False, num_workers=train_config.num_workers)
    test_loader = make_loader(test_dataset, batch_size=train_config.batch_size, shuffle=False, num_workers=train_config.num_workers)
    return train_loader, test_loader


def save_selected_feature_cache(
    feature_dir: str,
    split: str,
    features: torch.Tensor,
    labels: torch.Tensor,
    args: argparse.Namespace,
    layer_specs: list[LayerSpec],
    layer_selections: list[LayerSelection],
) -> None:
    output_dir = Path(feature_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_suffix = "-".join(str(spec.num_candidates) for spec in layer_specs)
    keep_suffix = "-".join(str(spec.keep_k) for spec in layer_specs)
    stride_suffix = "-".join(str(spec.stride) for spec in layer_specs)
    output_path = output_dir / (
        f"{split}_layerwise_selected_{args.backbone_kind}_"
        f"c{candidate_suffix}_k{keep_suffix}_s{stride_suffix}_"
        f"{args.selection_strategy}_seed{args.seed}.pt"
    )
    torch.save(
        {
            "features": features.to(torch.float16),
            "labels": labels,
            "split": split,
            "backbone_kind": f"layerwise_{args.backbone_kind}",
            "layer_specs": [spec.__dict__ for spec in layer_specs],
            "seed": args.seed,
            "global_feature_dim": int(features.shape[1]),
            "feature_family_masks": expand_feature_family_masks(layer_selections),
            "feature_family_names": {
                "overall": 1,
                "pairwise": 2,
                "one_vs_rest": 4,
            },
        },
        output_path,
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(args.seed)

    layer_specs = build_layer_specs(args.layer_candidates, args.layer_keeps, args.layer_strides)
    pairwise_extra_keeps = pad_optional_int_list(args.pairwise_extra_keeps, len(layer_specs))
    train_config = TrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        warmup_epochs=args.warmup_epochs,
        label_smoothing=args.label_smoothing,
    )
    device = torch.device(args.device)

    train_loader, test_loader = build_eval_loaders(train_config)
    selector = LayerwiseRandomFeatureStack(
        layer_specs=layer_specs,
        seed=args.seed,
        mode=args.backbone_kind,
        sine=SineConfig(),
    )
    train_inputs, train_labels = cache_loader_tensors(train_loader)
    test_inputs, test_labels = cache_loader_tensors(test_loader)

    layer_selections: list[LayerSelection] = []
    train_current_inputs = train_inputs
    train_pooled_parts = []
    for layer_index, spec in enumerate(layer_specs):
        selection, scores = selector.select_layer_from_cached_inputs(
            train_current_inputs,
            train_labels,
            layer_index=layer_index,
            num_classes=10,
            batch_size=train_config.batch_size,
            device=device,
            max_abs_correlation=args.max_abs_correlation,
            strategy=args.selection_strategy,
            pairwise_alpha=args.pairwise_alpha,
            multiclass_keep_fraction=args.multiclass_keep_fraction,
            pairwise_extra_keep_k=pairwise_extra_keeps[layer_index],
            pairwise_keep_fraction=args.pairwise_keep_fraction,
            ovr_keep_fraction=args.ovr_keep_fraction,
            shortlist_multiplier=args.selection_shortlist_multiplier,
        )
        layer_selections.append(selection)
        train_current_inputs, pooled_part = selector.advance_cached_inputs(
            train_current_inputs,
            layer_index=layer_index,
            propagated_indices=selection.propagated_indices,
            readout_indices=selection.readout_indices,
            batch_size=train_config.batch_size,
            device=device,
        )
        train_pooled_parts.append(pooled_part)
        best_score = float(scores[selection.propagated_indices[0]].item()) if selection.propagated_indices.numel() > 0 else 0.0
        print(
            f"layer={layer_index + 1} candidates={spec.num_candidates} keep_k={spec.keep_k} "
            f"propagated={selection.propagated_indices.numel()} "
            f"pairwise_aux={selection.readout_indices.numel()} best_score={best_score:.6f}"
        )

    if args.selection_path is not None:
        save_layerwise_selection(
            args.selection_path,
            layer_specs,
            layer_selections,
            metadata={
                "seed": args.seed,
                "backbone_kind": args.backbone_kind,
                "selection_strategy": args.selection_strategy,
                "pairwise_alpha": args.pairwise_alpha,
                "multiclass_keep_fraction": args.multiclass_keep_fraction,
                "pairwise_keep_fraction": args.pairwise_keep_fraction,
                "ovr_keep_fraction": args.ovr_keep_fraction,
                "selection_shortlist_multiplier": args.selection_shortlist_multiplier,
                "pairwise_extra_keeps": pairwise_extra_keeps,
            },
        )

    train_features = torch.cat(train_pooled_parts, dim=1)
    test_features = selector.extract_selected_features_from_cached_inputs(
        test_inputs,
        layer_selections,
        batch_size=train_config.batch_size,
        device=device,
    )
    if args.standardize_features:
        train_features, test_features = standardize_train_test(train_features, test_features)

    if args.save_feature_dir is not None:
        save_selected_feature_cache(args.save_feature_dir, "train", train_features, train_labels, args, layer_specs, layer_selections)
        save_selected_feature_cache(args.save_feature_dir, "test", test_features, test_labels, args, layer_specs, layer_selections)

    model_config = ModelConfig(
        backbone_kind=f"layerwise_{args.backbone_kind}",
        head_kind=args.head_kind,
        seed=args.seed,
        global_feature_dim=int(train_features.shape[1]),
    )
    head = make_head(args.head_kind, train_features.shape[1], model_config).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=train_config.label_smoothing)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in head.parameters() if parameter.requires_grad],
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    scheduler = make_scheduler(optimizer, train_config.epochs, train_config.warmup_epochs)

    head_train_loader = build_feature_loader(
        train_features,
        train_labels,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
    )
    head_test_loader = build_feature_loader(
        test_features,
        test_labels,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
    )

    print(
        f"device={device.type} feature_dim={train_features.shape[1]} "
        f"layers={len(layer_specs)} head={args.head_kind} standardize={args.standardize_features} "
        f"selection={args.selection_strategy} global_keeps={args.layer_keeps} "
        f"pairwise_aux={pairwise_extra_keeps}"
    )
    for epoch in range(train_config.epochs):
        train_loss, train_acc = run_epoch(
            head,
            head_train_loader,
            criterion,
            device,
            optimizer,
            l1_penalty=0.0,
            l2_penalty=0.0,
        )
        test_loss, test_acc = run_epoch(
            head,
            head_test_loader,
            criterion,
            device,
            optimizer=None,
            l1_penalty=0.0,
            l2_penalty=0.0,
        )
        print(
            f"epoch={epoch + 1} lr={scheduler.get_last_lr()[0]:.6f} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )
        scheduler.step()


if __name__ == "__main__":
    main()
