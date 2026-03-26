from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import TensorDataset

from .cache import default_topk_cache_path, load_topk_feature_cache, save_topk_feature_cache
from .config import SineConfig, TrainConfig
from .data import build_cifar10_datasets, cifar10_eval_transform, make_loader
from .layerwise_selection import LayerSelection, LayerSpec, LayerwiseRandomFeatureStack, cache_loader_tensors, save_layerwise_selection
from .pairwise import class_pairs, pairwise_targets, topk_refine_predictions
from .train_head import make_scheduler, standardize_train_test
from .train_layerwise_selected import build_layer_specs, parse_int_list, pad_optional_int_list


class TopKRefinementHead(nn.Module):
    def __init__(self, global_dim: int, pairwise_dim: int, num_classes: int) -> None:
        super().__init__()
        self.global_head = nn.Linear(global_dim, num_classes)
        self.pairwise_weight = nn.Parameter(torch.zeros(len(class_pairs(num_classes)), pairwise_dim))
        self.pairwise_bias = nn.Parameter(torch.zeros(len(class_pairs(num_classes))))
        nn.init.normal_(self.pairwise_weight, mean=0.0, std=1.0 / max(1, pairwise_dim) ** 0.5)

    def forward(self, global_features: torch.Tensor, pairwise_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        global_logits = self.global_head(global_features)
        pairwise_logits = torch.einsum("bpd,pd->bp", pairwise_features, self.pairwise_weight) + self.pairwise_bias
        return global_logits, pairwise_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a coarse global head with top-k pairwise refinement.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backbone-kind", choices=["strict_ones", "random_projection"], default="random_projection")
    parser.add_argument("--layer-candidates", type=parse_int_list, default=parse_int_list("1024,1024,1024"))
    parser.add_argument("--layer-keeps", type=parse_int_list, default=parse_int_list("128,128,128"))
    parser.add_argument("--pairwise-per-pair-keeps", type=parse_int_list, default=parse_int_list("0,4,8"))
    parser.add_argument("--layer-strides", type=parse_int_list, default=parse_int_list("1,2,2"))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--pairwise-loss-weight", type=float, default=1.0)
    parser.add_argument("--pairwise-scale", type=float, default=0.5)
    parser.add_argument("--top-k-refine", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-abs-correlation", type=float, default=0.95)
    parser.add_argument("--standardize-features", action="store_true")
    parser.add_argument("--selection-path", type=str)
    parser.add_argument("--save-feature-dir", type=str)
    parser.add_argument("--load-feature-dir", type=str)
    parser.add_argument("--extract-features-only", action="store_true")
    parser.add_argument(
        "--device",
        default="cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu",
    )
    return parser.parse_args()


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


def standardize_global_and_pairwise(
    train_global: torch.Tensor,
    test_global: torch.Tensor,
    train_pairwise: torch.Tensor,
    test_pairwise: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    train_global_std, test_global_std = standardize_train_test(train_global, test_global)
    if train_pairwise.numel() == 0:
        return train_global_std, test_global_std, train_pairwise.to(torch.float32), test_pairwise.to(torch.float32)

    train_pairwise_float = train_pairwise.to(torch.float32)
    test_pairwise_float = test_pairwise.to(torch.float32)
    mean = train_pairwise_float.mean(dim=0, keepdim=True)
    std = train_pairwise_float.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (
        train_global_std,
        test_global_std,
        (train_pairwise_float - mean) / std,
        (test_pairwise_float - mean) / std,
    )


def build_refinement_loader(
    global_features: torch.Tensor,
    pairwise_features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
):
    pair_targets, pair_mask = pairwise_targets(labels, num_classes=10)
    dataset = TensorDataset(global_features, pairwise_features, labels, pair_targets, pair_mask)
    return make_loader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def topk_cache_metadata(
    args: argparse.Namespace,
    layer_specs: list[LayerSpec],
    pairwise_per_pair_keeps: list[int],
) -> dict:
    return {
        "backbone_kind": args.backbone_kind,
        "seed": args.seed,
        "layer_specs": [spec.__dict__ for spec in layer_specs],
        "pairwise_per_pair_keeps": pairwise_per_pair_keeps,
        "standardized_features": False,
    }


def topk_cache_paths(
    directory: str,
    args: argparse.Namespace,
    layer_specs: list[LayerSpec],
    pairwise_per_pair_keeps: list[int],
) -> tuple[Path, Path]:
    train_path = default_topk_cache_path(
        directory=directory,
        split="train",
        backbone_kind=args.backbone_kind,
        layer_specs=layer_specs,
        pairwise_per_pair_keeps=pairwise_per_pair_keeps,
        seed=args.seed,
        standardized=False,
    )
    test_path = default_topk_cache_path(
        directory=directory,
        split="test",
        backbone_kind=args.backbone_kind,
        layer_specs=layer_specs,
        pairwise_per_pair_keeps=pairwise_per_pair_keeps,
        seed=args.seed,
        standardized=False,
    )
    return train_path, test_path


def pairwise_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    losses = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    masked = losses * mask
    return masked.sum() / mask.sum().clamp_min(1.0)


def run_epoch(
    model: TopKRefinementHead,
    loader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    pairwise_loss_weight: float,
    pairwise_scale: float,
    top_k_refine: int,
) -> tuple[float, float]:
    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0
    is_training = optimizer is not None
    model.train(is_training)

    grad_context = torch.enable_grad() if is_training else torch.inference_mode()
    with grad_context:
        for global_features, pairwise_features, labels, pair_targets, pair_mask in loader:
            global_features = global_features.to(device, non_blocking=True).to(torch.float32)
            pairwise_features = pairwise_features.to(device, non_blocking=True).to(torch.float32)
            labels = labels.to(device, non_blocking=True)
            pair_targets = pair_targets.to(device, non_blocking=True)
            pair_mask = pair_mask.to(device, non_blocking=True)

            global_logits, pairwise_logits = model(global_features, pairwise_features)
            loss = criterion(global_logits, labels) + pairwise_loss_weight * pairwise_loss(pairwise_logits, pair_targets, pair_mask)

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            predictions = topk_refine_predictions(
                global_logits.detach(),
                pairwise_logits.detach(),
                num_classes=10,
                top_k=top_k_refine,
                pairwise_scale=pairwise_scale,
            )
            total_loss += float(loss.item())
            total_acc += float((predictions == labels).float().mean().item())
            total_batches += 1

    return total_loss / total_batches, total_acc / total_batches


def main() -> None:
    args = parse_args()
    if args.load_feature_dir is not None and args.save_feature_dir is not None:
        raise ValueError("Choose either --load-feature-dir or --save-feature-dir, not both.")
    if args.extract_features_only and args.save_feature_dir is None:
        raise ValueError("--extract-features-only requires --save-feature-dir.")
    torch.manual_seed(args.seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(args.seed)

    layer_specs = build_layer_specs(args.layer_candidates, args.layer_keeps, args.layer_strides)
    pairwise_per_pair_keeps = pad_optional_int_list(args.pairwise_per_pair_keeps, len(layer_specs))
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
    if args.load_feature_dir is not None:
        train_cache_path, test_cache_path = topk_cache_paths(args.load_feature_dir, args, layer_specs, pairwise_per_pair_keeps)
        train_global, train_pairwise, train_labels, train_metadata = load_topk_feature_cache(train_cache_path)
        test_global, test_pairwise, test_labels, test_metadata = load_topk_feature_cache(test_cache_path)
        if train_metadata["pairwise_per_pair_keeps"] != pairwise_per_pair_keeps:
            raise ValueError("Loaded train cache does not match the requested pairwise-per-pair schedule.")
        if test_metadata["pairwise_per_pair_keeps"] != pairwise_per_pair_keeps:
            raise ValueError("Loaded test cache does not match the requested pairwise-per-pair schedule.")
        print(f"loaded_feature_cache train={train_cache_path} test={test_cache_path}")
    else:
        train_loader, test_loader = build_eval_loaders(train_config)
        train_inputs, train_labels = cache_loader_tensors(train_loader)
        test_inputs, test_labels = cache_loader_tensors(test_loader)

        selector = LayerwiseRandomFeatureStack(
            layer_specs=layer_specs,
            seed=args.seed,
            mode=args.backbone_kind,
            sine=SineConfig(),
        )

        layer_selections: list[LayerSelection] = []
        train_current_inputs = train_inputs
        for layer_index, spec in enumerate(layer_specs):
            selection, scores = selector.select_layer_from_cached_inputs(
                train_current_inputs,
                train_labels,
                layer_index=layer_index,
                num_classes=10,
                batch_size=train_config.batch_size,
                device=device,
                max_abs_correlation=args.max_abs_correlation,
                strategy="global_pairwise_refinement",
                pairwise_extra_keep_k=pairwise_per_pair_keeps[layer_index],
            )
            layer_selections.append(selection)
            train_current_inputs, _ = selector.advance_cached_inputs(
                train_current_inputs,
                layer_index=layer_index,
                propagated_indices=selection.propagated_indices,
                readout_indices=None,
                batch_size=train_config.batch_size,
                device=device,
            )
            best_score = float(scores[selection.propagated_indices[0]].item()) if selection.propagated_indices.numel() > 0 else 0.0
            per_pair = 0 if selection.pairwise_indices is None else int(selection.pairwise_indices.shape[1])
            print(
                f"layer={layer_index + 1} candidates={spec.num_candidates} keep_k={spec.keep_k} "
                f"propagated={selection.propagated_indices.numel()} pairwise_per_pair={per_pair} best_score={best_score:.6f}"
            )

        if args.selection_path is not None:
            save_layerwise_selection(
                args.selection_path,
                layer_specs,
                layer_selections,
                metadata={
                    "seed": args.seed,
                    "backbone_kind": args.backbone_kind,
                    "selection_strategy": "global_pairwise_refinement",
                    "pairwise_per_pair_keeps": pairwise_per_pair_keeps,
                },
            )

        train_global, train_pairwise = selector.extract_global_and_pairwise_features_from_cached_inputs(
            train_inputs,
            layer_selections,
            batch_size=train_config.batch_size,
            device=device,
        )
        test_global, test_pairwise = selector.extract_global_and_pairwise_features_from_cached_inputs(
            test_inputs,
            layer_selections,
            batch_size=train_config.batch_size,
            device=device,
        )
        if args.save_feature_dir is not None:
            cache_metadata = topk_cache_metadata(args, layer_specs, pairwise_per_pair_keeps)
            train_cache_path, test_cache_path = topk_cache_paths(args.save_feature_dir, args, layer_specs, pairwise_per_pair_keeps)
            save_topk_feature_cache(
                output_path=train_cache_path,
                global_features=train_global.to(torch.float16),
                pairwise_features=train_pairwise.to(torch.float16),
                labels=train_labels,
                metadata=cache_metadata,
                split="train",
            )
            save_topk_feature_cache(
                output_path=test_cache_path,
                global_features=test_global.to(torch.float16),
                pairwise_features=test_pairwise.to(torch.float16),
                labels=test_labels,
                metadata=cache_metadata,
                split="test",
            )
            print(f"saved_feature_cache train={train_cache_path} test={test_cache_path}")
            if args.extract_features_only:
                return

    if args.standardize_features:
        train_global, test_global, train_pairwise, test_pairwise = standardize_global_and_pairwise(
            train_global,
            test_global,
            train_pairwise,
            test_pairwise,
        )

    pairwise_dim = 0 if train_pairwise.numel() == 0 else int(train_pairwise.shape[2])
    model = TopKRefinementHead(train_global.shape[1], pairwise_dim, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=train_config.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
    scheduler = make_scheduler(optimizer, train_config.epochs, train_config.warmup_epochs)

    train_refine_loader = build_refinement_loader(
        train_global,
        train_pairwise,
        train_labels,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
    )
    test_refine_loader = build_refinement_loader(
        test_global,
        test_pairwise,
        test_labels,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
    )

    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(
        f"trainable_parameters={trainable_parameters} global_dim={train_global.shape[1]} "
        f"pairwise_dim={pairwise_dim} top_k_refine={args.top_k_refine}"
    )
    for epoch in range(train_config.epochs):
        train_loss, train_acc = run_epoch(
            model,
            train_refine_loader,
            criterion,
            device,
            optimizer,
            pairwise_loss_weight=args.pairwise_loss_weight,
            pairwise_scale=args.pairwise_scale,
            top_k_refine=args.top_k_refine,
        )
        test_loss, test_acc = run_epoch(
            model,
            test_refine_loader,
            criterion,
            device,
            optimizer=None,
            pairwise_loss_weight=args.pairwise_loss_weight,
            pairwise_scale=args.pairwise_scale,
            top_k_refine=args.top_k_refine,
        )
        print(
            f"epoch={epoch + 1} lr={scheduler.get_last_lr()[0]:.6f} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )
        scheduler.step()


if __name__ == "__main__":
    main()
