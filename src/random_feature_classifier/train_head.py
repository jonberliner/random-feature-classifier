from __future__ import annotations

import argparse
import math

import torch
from torch import nn

from .cache import load_feature_cache
from .config import ModelConfig, TrainConfig
from .data import build_feature_loader
from .heads import make_head
from .selection import load_selection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a head on cached fixed features.")
    parser.add_argument("--head-kind", choices=["linear", "scalar"], default="linear")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--l1-penalty", type=float, default=0.0)
    parser.add_argument("--l2-penalty", type=float, default=0.0)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--test-cache", required=True)
    parser.add_argument("--selection-path")
    parser.add_argument("--top-k", type=int, default=0)
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


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return float((predictions == labels).float().mean().item())


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(epoch_index: int) -> float:
        current_epoch = epoch_index + 1
        if current_epoch <= warmup_epochs:
            return current_epoch / max(1, warmup_epochs)
        progress = (current_epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def run_epoch(
    head: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    l1_penalty: float,
    l2_penalty: float,
) -> tuple[float, float]:
    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0
    is_training = optimizer is not None
    head.train(is_training)

    grad_context = torch.enable_grad() if is_training else torch.inference_mode()
    with grad_context:
        for features, labels in loader:
            features = features.to(device, non_blocking=True).to(torch.float32)
            labels = labels.to(device, non_blocking=True)
            logits = head(features)
            loss = criterion(logits, labels)
            if l1_penalty > 0.0 or l2_penalty > 0.0:
                regularization = regularization_penalty(head, l1_penalty, l2_penalty)
                loss = loss + regularization

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_loss += float(loss.item())
            total_acc += accuracy(logits.detach(), labels)
            total_batches += 1

    return total_loss / total_batches, total_acc / total_batches


def infer_feature_dim(metadata: dict, features: torch.Tensor) -> int:
    stored_dim = int(metadata.get("global_feature_dim", 0))
    return stored_dim if stored_dim > 0 else int(features.shape[1])


def regularization_penalty(head: nn.Module, l1_penalty: float, l2_penalty: float) -> torch.Tensor:
    penalty = torch.zeros((), device=next(head.parameters()).device)
    for parameter in head.parameters():
        if not parameter.requires_grad:
            continue
        if l1_penalty > 0.0:
            penalty = penalty + l1_penalty * parameter.abs().sum()
        if l2_penalty > 0.0:
            penalty = penalty + l2_penalty * parameter.square().sum()
    return penalty


def apply_feature_selection(
    train_features: torch.Tensor,
    test_features: torch.Tensor,
    selection_path: str | None,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if selection_path is None:
        return train_features, test_features

    selection = load_selection(selection_path)
    ranked_indices = selection["ranked_indices"].to(torch.int64)
    if top_k > 0:
        ranked_indices = ranked_indices[:top_k]
    return train_features[:, ranked_indices], test_features[:, ranked_indices]


def standardize_train_test(
    train_features: torch.Tensor,
    test_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    train_float = train_features.to(torch.float32)
    test_float = test_features.to(torch.float32)
    mean = train_float.mean(dim=0, keepdim=True)
    std = train_float.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (train_float - mean) / std, (test_float - mean) / std


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(args.seed)

    train_features, train_labels, train_meta = load_feature_cache(args.train_cache)
    test_features, test_labels, _ = load_feature_cache(args.test_cache)
    if train_features.shape[1] != test_features.shape[1]:
        raise ValueError("Train and test caches have different feature dimensions.")

    train_features, test_features = apply_feature_selection(
        train_features,
        test_features,
        selection_path=args.selection_path,
        top_k=args.top_k,
    )
    if args.standardize_features:
        train_features, test_features = standardize_train_test(train_features, test_features)

    feature_dim = infer_feature_dim(train_meta, train_features)
    model_config = ModelConfig(
        backbone_kind=str(train_meta.get("backbone_kind", "strict_ones")),
        head_kind=args.head_kind,
        seed=args.seed,
        width_multiplier=int(train_meta.get("width_multiplier", 2)),
        tap_stages=tuple(train_meta.get("tap_stages", (1, 2, 3, 4))),
        global_feature_dim=feature_dim,
    )
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
    head = make_head(args.head_kind, train_features.shape[1], model_config).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=train_config.label_smoothing)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in head.parameters() if parameter.requires_grad],
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    scheduler = make_scheduler(optimizer, train_config.epochs, train_config.warmup_epochs)

    train_loader = build_feature_loader(
        train_features,
        train_labels,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
    )
    test_loader = build_feature_loader(
        test_features,
        test_labels,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
    )

    trainable_parameters = sum(parameter.numel() for parameter in head.parameters() if parameter.requires_grad)
    print(f"trainable_parameters={trainable_parameters}")
    print(
        f"device={device.type} batch_size={train_config.batch_size} "
        f"feature_dim={train_features.shape[1]} learning_rate={train_config.learning_rate} "
        f"l1_penalty={args.l1_penalty} l2_penalty={args.l2_penalty} "
        f"top_k={args.top_k if args.top_k > 0 else train_features.shape[1]} "
        f"standardize={args.standardize_features}"
    )

    for epoch in range(train_config.epochs):
        train_loss, train_acc = run_epoch(
            head,
            train_loader,
            criterion,
            device,
            optimizer,
            l1_penalty=args.l1_penalty,
            l2_penalty=args.l2_penalty,
        )
        test_loss, test_acc = run_epoch(
            head,
            test_loader,
            criterion,
            device,
            optimizer=None,
            l1_penalty=args.l1_penalty,
            l2_penalty=args.l2_penalty,
        )
        print(
            f"epoch={epoch + 1} "
            f"lr={scheduler.get_last_lr()[0]:.6f} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )
        scheduler.step()


if __name__ == "__main__":
    main()
