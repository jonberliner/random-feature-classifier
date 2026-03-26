from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, transforms

from .config import TrainConfig


def cifar10_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_transform, test_transform


def cifar10_eval_transform() -> transforms.Compose:
    _, test_transform = cifar10_transforms()
    return test_transform


def build_cifar10_datasets(
    config: TrainConfig,
    train_transform: transforms.Compose | None = None,
    test_transform: transforms.Compose | None = None,
) -> tuple[Dataset, Dataset]:
    train_transform = train_transform or cifar10_transforms()[0]
    test_transform = test_transform or cifar10_transforms()[1]
    train_dataset = datasets.CIFAR10(
        root=config.data_root,
        train=True,
        download=True,
        transform=train_transform,
    )
    test_dataset = datasets.CIFAR10(
        root=config.data_root,
        train=False,
        download=True,
        transform=test_transform,
    )
    return train_dataset, test_dataset


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
    )


def build_cifar10_loaders(config: TrainConfig) -> tuple[DataLoader, DataLoader]:
    train_dataset, test_dataset = build_cifar10_datasets(config)
    train_loader = make_loader(train_dataset, config.batch_size, True, config.num_workers)
    test_loader = make_loader(test_dataset, config.batch_size, False, config.num_workers)
    return train_loader, test_loader


def build_feature_loader(
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = TensorDataset(features, labels)
    return make_loader(dataset, batch_size, shuffle, num_workers)
