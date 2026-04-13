"""
CIFAR-10 / CIFAR-100 数据：增强、Dataset、DataLoader（可复现）。
"""
from __future__ import annotations

import os
import random
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from cifar import CIFAR10Dataset, CIFAR100Dataset


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_transforms(dataset_name: str) -> Tuple[transforms.Compose, transforms.Compose, Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
    if dataset_name == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif dataset_name == "cifar100":
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    base_train = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]

    if dataset_name == "cifar100":
        base_train += [
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value="random"),
        ]

    base_train += [transforms.Normalize(mean, std)]
    transform_train = transforms.Compose(base_train)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return transform_train, transform_test, (mean, std)


def build_datasets(dataset_name: str, data_root: str, transform_train, transform_test):
    if dataset_name == "cifar10":
        trainset = CIFAR10Dataset(
            root=os.path.join(data_root, "cifar-10-batches-py"),
            train=True,
            transform=transform_train,
        )
        testset = CIFAR10Dataset(
            root=os.path.join(data_root, "cifar-10-batches-py"),
            train=False,
            transform=transform_test,
        )
        num_classes = 10
    elif dataset_name == "cifar100":
        root100 = os.path.join(data_root, "cifar-100-python")
        trainset = CIFAR100Dataset(root=root100, train=True, transform=transform_train)
        testset = CIFAR100Dataset(root=root100, train=False, transform=transform_test)
        num_classes = 100
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return trainset, testset, num_classes


def build_test_dataset(dataset_name: str, data_root: str, transform_test):
    """仅测试集（不参与 train/val 划分）。"""
    if dataset_name == "cifar10":
        testset = CIFAR10Dataset(
            root=os.path.join(data_root, "cifar-10-batches-py"),
            train=False,
            transform=transform_test,
        )
        num_classes = 10
    elif dataset_name == "cifar100":
        root100 = os.path.join(data_root, "cifar-100-python")
        testset = CIFAR100Dataset(root=root100, train=False, transform=transform_test)
        num_classes = 100
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return testset, num_classes


def split_train_val_subsets(
    dataset_name: str,
    data_root: str,
    transform_train,
    transform_val,
    val_ratio: float,
    seed: int,
) -> Tuple[Subset, Subset, int]:
    """
    将官方 train 划分为 train_sub（训练增强）与 val_sub（与 test 相同的确定性变换）。
    划分索引由 seed 固定，可复现。
    """
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio 必须在 (0, 1) 内，当前为 {val_ratio}")

    if dataset_name == "cifar10":
        root = os.path.join(data_root, "cifar-10-batches-py")
        ds_tr = CIFAR10Dataset(root, train=True, transform=transform_train)
        ds_va = CIFAR10Dataset(root, train=True, transform=transform_val)
        num_classes = 10
    elif dataset_name == "cifar100":
        root100 = os.path.join(data_root, "cifar-100-python")
        ds_tr = CIFAR100Dataset(root=root100, train=True, transform=transform_train)
        ds_va = CIFAR100Dataset(root=root100, train=True, transform=transform_val)
        num_classes = 100
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    n = len(ds_tr)
    n_val = int(round(n * val_ratio))
    n_val = max(1, min(n - 1, n_val))

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    val_indices = perm[:n_val]
    train_indices = perm[n_val:]

    return Subset(ds_tr, train_indices), Subset(ds_va, val_indices), num_classes


def build_dataloaders(
    trainset,
    testset,
    batch_size: int,
    test_batch_size: int,
    num_workers: int,
    seed: int,
):
    common_kwargs: Dict[str, object] = dict(pin_memory=True)
    if num_workers > 0:
        common_kwargs.update(dict(prefetch_factor=2, persistent_workers=True, num_workers=num_workers))
    else:
        common_kwargs.update(dict(num_workers=0))

    g = torch.Generator()
    g.manual_seed(seed)

    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        **common_kwargs,
        worker_init_fn=seed_worker,
        generator=g,
    )
    testloader = DataLoader(
        testset,
        batch_size=test_batch_size,
        shuffle=False,
        **common_kwargs,
        worker_init_fn=seed_worker,
        generator=g,
    )
    return trainloader, testloader


def build_train_val_test_loaders(
    train_set,
    val_set,
    test_set,
    batch_size: int,
    test_batch_size: int,
    num_workers: int,
    seed: int,
):
    """train shuffle=True；val/test shuffle=False。"""
    common_kwargs: Dict[str, object] = dict(pin_memory=True)
    if num_workers > 0:
        common_kwargs.update(dict(prefetch_factor=2, persistent_workers=True, num_workers=num_workers))
    else:
        common_kwargs.update(dict(num_workers=0))

    g = torch.Generator()
    g.manual_seed(seed)

    trainloader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        **common_kwargs,
        worker_init_fn=seed_worker,
        generator=g,
    )
    valloader = DataLoader(
        val_set,
        batch_size=test_batch_size,
        shuffle=False,
        **common_kwargs,
        worker_init_fn=seed_worker,
        generator=g,
    )
    testloader = DataLoader(
        test_set,
        batch_size=test_batch_size,
        shuffle=False,
        **common_kwargs,
        worker_init_fn=seed_worker,
        generator=g,
    )
    return trainloader, valloader, testloader
