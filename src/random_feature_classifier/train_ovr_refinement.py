from __future__ import annotations

import argparse

import torch
from torch import nn
from torch.utils.data import TensorDataset

from .cache import load_matrix_feature_cache
from .config import TrainConfig
from .data import make_loader
from .ovr import one_vs_rest_targets, refine_with_ovr_predictions
from .train_head import make_scheduler, standardize_train_test


class OvrRefinementHead(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.coarse_head = nn.Linear(feature_dim, num_classes)
        self.ovr_head = nn.Linear(feature_dim, num_classes)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.coarse_head(features), self.ovr_head(features)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a coarse multiclass head with one-vs-rest residual refinement.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--test-cache", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--standardize-features", action="store_true")
    parser.add_argument("--residual-loss-weight", type=float, default=0.5)
    parser.add_argument("--residual-scale", type=float, default=0.5)
    parser.add_argument("--residual-positive-weight", type=float, default=9.0)
    parser.add_argument("--top-k-refine", type=int, default=3)
    parser.add_argument("--refine-margin-threshold", type=float, default=-1.0)
    parser.add_argument(
        "--device",
        default="cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu",
    )
    return parser.parse_args()


def build_loader(
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    num_classes: int,
):
    targets = one_vs_rest_targets(labels, num_classes=num_classes)
    dataset = TensorDataset(features, labels, targets)
    return make_loader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def run_epoch(
    model: OvrRefinementHead,
    loader,
    coarse_criterion: nn.Module,
    residual_criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    residual_loss_weight: float,
    residual_scale: float,
    top_k_refine: int,
    margin_threshold: float,
) -> tuple[float, float, float]:
    total_loss = 0.0
    total_coarse_acc = 0.0
    total_refined_acc = 0.0
    total_batches = 0
    is_training = optimizer is not None
    model.train(is_training)

    grad_context = torch.enable_grad() if is_training else torch.inference_mode()
    with grad_context:
        for features, labels, targets in loader:
            features = features.to(device, non_blocking=True).to(torch.float32)
            labels = labels.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            coarse_logits, residual_logits = model(features)
            loss = coarse_criterion(coarse_logits, labels) + residual_loss_weight * residual_criterion(residual_logits, targets)

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            coarse_predictions = coarse_logits.argmax(dim=1)
            refined_predictions = refine_with_ovr_predictions(
                coarse_logits.detach(),
                residual_logits.detach(),
                top_k=top_k_refine,
                residual_scale=residual_scale,
                margin_threshold=margin_threshold,
            )
            total_loss += float(loss.item())
            total_coarse_acc += float((coarse_predictions == labels).float().mean().item())
            total_refined_acc += float((refined_predictions == labels).float().mean().item())
            total_batches += 1

    return total_loss / total_batches, total_coarse_acc / total_batches, total_refined_acc / total_batches


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(args.seed)

    train_features, train_labels, train_meta = load_matrix_feature_cache(args.train_cache)
    test_features, test_labels, _ = load_matrix_feature_cache(args.test_cache)
    if train_features.shape[1] != test_features.shape[1]:
        raise ValueError("Train and test caches have different feature dimensions.")

    if args.standardize_features:
        train_features, test_features = standardize_train_test(train_features, test_features)

    num_classes = int(train_meta.get("num_classes", 10))
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

    model = OvrRefinementHead(train_features.shape[1], num_classes=num_classes).to(device)
    coarse_criterion = nn.CrossEntropyLoss(label_smoothing=train_config.label_smoothing)
    pos_weight = torch.full((num_classes,), args.residual_positive_weight, dtype=torch.float32, device=device)
    residual_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
    scheduler = make_scheduler(optimizer, train_config.epochs, train_config.warmup_epochs)

    train_loader = build_loader(
        train_features,
        train_labels,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        num_classes=num_classes,
    )
    test_loader = build_loader(
        test_features,
        test_labels,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        num_classes=num_classes,
    )

    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(
        f"trainable_parameters={trainable_parameters} feature_dim={train_features.shape[1]} "
        f"top_k_refine={args.top_k_refine} residual_scale={args.residual_scale:.3f} "
        f"margin_threshold={args.refine_margin_threshold:.3f}"
    )

    for epoch in range(train_config.epochs):
        train_loss, train_coarse_acc, train_refined_acc = run_epoch(
            model,
            train_loader,
            coarse_criterion,
            residual_criterion,
            device,
            optimizer,
            residual_loss_weight=args.residual_loss_weight,
            residual_scale=args.residual_scale,
            top_k_refine=args.top_k_refine,
            margin_threshold=args.refine_margin_threshold,
        )
        test_loss, test_coarse_acc, test_refined_acc = run_epoch(
            model,
            test_loader,
            coarse_criterion,
            residual_criterion,
            device,
            optimizer=None,
            residual_loss_weight=args.residual_loss_weight,
            residual_scale=args.residual_scale,
            top_k_refine=args.top_k_refine,
            margin_threshold=args.refine_margin_threshold,
        )
        print(
            f"epoch={epoch + 1} lr={scheduler.get_last_lr()[0]:.6f} "
            f"train_loss={train_loss:.4f} train_coarse_acc={train_coarse_acc:.4f} "
            f"train_refined_acc={train_refined_acc:.4f} test_loss={test_loss:.4f} "
            f"test_coarse_acc={test_coarse_acc:.4f} test_refined_acc={test_refined_acc:.4f}"
        )
        scheduler.step()


if __name__ == "__main__":
    main()
