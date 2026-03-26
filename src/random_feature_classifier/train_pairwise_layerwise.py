from __future__ import annotations

import argparse

import torch
from torch import nn
from torch.utils.data import TensorDataset

from .config import SineConfig, TrainConfig
from .data import build_cifar10_datasets, cifar10_eval_transform, make_loader
from .layerwise_selection import LayerwiseRandomFeatureStack, cache_loader_tensors, load_layerwise_selection
from .pairwise import class_pairs, pairwise_targets, pairwise_vote_predictions
from .train_head import make_scheduler, standardize_train_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train pairwise one-vs-one heads on a shared layerwise-selected extractor.")
    parser.add_argument("--selection-path", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--standardize-features", action="store_true")
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


def build_pairwise_loader(
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> torch.utils.data.DataLoader:
    targets, mask = pairwise_targets(labels, num_classes=10)
    dataset = TensorDataset(features, labels, targets, mask)
    return make_loader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def pairwise_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    losses = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    masked = losses * mask
    return masked.sum() / mask.sum().clamp_min(1.0)


def run_epoch(
    head: nn.Module,
    loader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0
    is_training = optimizer is not None
    head.train(is_training)

    grad_context = torch.enable_grad() if is_training else torch.inference_mode()
    with grad_context:
        for features, labels, targets, mask in loader:
            features = features.to(device, non_blocking=True).to(torch.float32)
            labels = labels.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            logits = head(features)
            loss = pairwise_loss(logits, targets, mask)

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            predictions = pairwise_vote_predictions(logits.detach(), num_classes=10)
            total_loss += float(loss.item())
            total_acc += float((predictions == labels).float().mean().item())
            total_batches += 1

    return total_loss / total_batches, total_acc / total_batches


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(args.seed)

    selection = load_layerwise_selection(args.selection_path)
    train_config = TrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        warmup_epochs=args.warmup_epochs,
    )
    device = torch.device(args.device)

    train_loader, test_loader = build_eval_loaders(train_config)
    train_inputs, train_labels = cache_loader_tensors(train_loader)
    test_inputs, test_labels = cache_loader_tensors(test_loader)

    selector = LayerwiseRandomFeatureStack(
        layer_specs=selection["layer_specs"],
        seed=int(selection.get("seed", args.seed)),
        mode=str(selection.get("backbone_kind", "random_projection")),
        sine=SineConfig(),
    )
    train_features = selector.extract_selected_features_from_cached_inputs(
        train_inputs,
        selection["selected_indices"],
        batch_size=train_config.batch_size,
        device=device,
    )
    test_features = selector.extract_selected_features_from_cached_inputs(
        test_inputs,
        selection["selected_indices"],
        batch_size=train_config.batch_size,
        device=device,
    )
    if args.standardize_features:
        train_features, test_features = standardize_train_test(train_features, test_features)

    head = nn.Linear(train_features.shape[1], len(class_pairs(10))).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
    scheduler = make_scheduler(optimizer, train_config.epochs, train_config.warmup_epochs)

    pair_train_loader = build_pairwise_loader(
        train_features,
        train_labels,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
    )
    pair_test_loader = build_pairwise_loader(
        test_features,
        test_labels,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
    )

    trainable_parameters = sum(parameter.numel() for parameter in head.parameters() if parameter.requires_grad)
    print(f"trainable_parameters={trainable_parameters}")
    print(
        f"device={device.type} feature_dim={train_features.shape[1]} num_pairs={len(class_pairs(10))} "
        f"selection_path={args.selection_path} standardize={args.standardize_features}"
    )
    for epoch in range(train_config.epochs):
        train_loss, train_acc = run_epoch(head, pair_train_loader, device, optimizer)
        test_loss, test_acc = run_epoch(head, pair_test_loader, device, optimizer=None)
        print(
            f"epoch={epoch + 1} lr={scheduler.get_last_lr()[0]:.6f} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )
        scheduler.step()


if __name__ == "__main__":
    main()
