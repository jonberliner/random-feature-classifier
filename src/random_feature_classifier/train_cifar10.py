from __future__ import annotations

import argparse
import math

import torch
from torch import nn

from .config import ModelConfig, TrainConfig
from .data import build_cifar10_loaders
from .model import make_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fixed-feature CIFAR-10 models.")
    parser.add_argument("--backbone-kind", choices=["strict_ones", "random_projection"], default="strict_ones")
    parser.add_argument("--head-kind", choices=["linear", "scalar"], default="linear")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width-multiplier", type=int, default=2)
    parser.add_argument("--tap-stages", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--global-feature-dim", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
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


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return float((predictions == labels).float().mean().item())


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0
    is_training = optimizer is not None
    model.train(is_training)

    grad_context = torch.enable_grad() if is_training else torch.inference_mode()
    with grad_context:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            images = images.contiguous(memory_format=torch.channels_last)
            logits = model(images)
            loss = criterion(logits, labels)

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_loss += float(loss.item())
            total_acc += accuracy(logits.detach(), labels)
            total_batches += 1

    return total_loss / total_batches, total_acc / total_batches


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


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(args.seed)

    model_config = ModelConfig(
        backbone_kind=args.backbone_kind,
        head_kind=args.head_kind,
        seed=args.seed,
        width_multiplier=args.width_multiplier,
        tap_stages=tuple(args.tap_stages),
        global_feature_dim=args.global_feature_dim,
    )
    train_config = TrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        data_root=args.data_root,
        num_workers=args.num_workers,
        warmup_epochs=args.warmup_epochs,
        label_smoothing=args.label_smoothing,
    )

    device = torch.device(args.device)
    model = make_model(model_config).to(device=device, memory_format=torch.channels_last)
    criterion = nn.CrossEntropyLoss(label_smoothing=train_config.label_smoothing)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    scheduler = make_scheduler(optimizer, train_config.epochs, train_config.warmup_epochs)

    train_loader, test_loader = build_cifar10_loaders(train_config)
    print(f"trainable_parameters={sum(parameter.numel() for parameter in parameters)}")
    print(
        f"device={device.type} batch_size={train_config.batch_size} "
        f"learning_rate={train_config.learning_rate} warmup_epochs={train_config.warmup_epochs}"
    )

    for epoch in range(train_config.epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        test_loss, test_acc = run_epoch(model, test_loader, criterion, device, optimizer=None)
        print(
            f"epoch={epoch + 1} "
            f"lr={scheduler.get_last_lr()[0]:.6f} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )
        scheduler.step()


if __name__ == "__main__":
    main()
