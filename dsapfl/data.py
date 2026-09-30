"""Datasets and reproducible client partitions for DSA-PFL."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms

from .config import Config


@dataclass(frozen=True)
class Partition:
    train_indices: tuple[tuple[int, ...], ...]
    test_indices: tuple[tuple[int, ...], ...]
    digest: str
    overlap_count: int
    unassigned_count: int


class TinyImageNetSplit(Dataset[Any]):
    def __init__(self, root: Path, train: bool, transform: Any) -> None:
        self.transform = transform
        wnids_path = root / "wnids.txt"
        if not wnids_path.is_file():
            raise FileNotFoundError(f"Tiny-ImageNet wnids.txt not found: {wnids_path}")
        wnids = [line.strip() for line in wnids_path.read_text().splitlines() if line.strip()]
        class_to_idx = {wnid: index for index, wnid in enumerate(wnids)}
        self.samples: list[tuple[Path, int]] = []
        if train:
            for wnid in wnids:
                for path in sorted((root / "train" / wnid / "images").glob("*.JPEG")):
                    self.samples.append((path, class_to_idx[wnid]))
        else:
            annotations = root / "val" / "val_annotations.txt"
            image_dir = root / "val" / "images"
            if not annotations.is_file():
                raise FileNotFoundError(f"validation annotations not found: {annotations}")
            for line in annotations.read_text().splitlines():
                fields = line.split("\t")
                if len(fields) >= 2 and fields[1] in class_to_idx:
                    self.samples.append((image_dir / fields[0], class_to_idx[fields[1]]))
        if not self.samples:
            raise RuntimeError(f"no Tiny-ImageNet images found under {root}")
        self.targets = [target for _, target in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, target = self.samples[index]
        with Image.open(path) as image:
            value = image.convert("RGB")
        return self.transform(value), target


def load_dataset(config: Config) -> tuple[Dataset[Any], np.ndarray]:
    """Load the historical merged-pool protocol used by the completed runs."""

    if not config.merge_official_splits:
        raise NotImplementedError(
            "This release currently freezes the completed merged-pool protocol; "
            "implement and validate a separate official-test protocol before use."
        )
    if config.dataset == "tinyimagenet":
        root = config.data_root.expanduser().resolve()
        if root.name != "tiny-imagenet-200":
            root = root / "tiny-imagenet-200"
        transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
        train = TinyImageNetSplit(root, True, transform)
        test = TinyImageNetSplit(root, False, transform)
    else:
        if config.dataset == "fashionmnist":
            transform = transforms.Compose(
                [transforms.Grayscale(num_output_channels=3), transforms.ToTensor()]
            )
            dataset_type = datasets.FashionMNIST
        elif config.dataset == "cifar10":
            transform = transforms.ToTensor()
            dataset_type = datasets.CIFAR10
        else:
            transform = transforms.ToTensor()
            dataset_type = datasets.CIFAR100
        train = dataset_type(
            root=str(config.data_root), train=True, download=config.download, transform=transform
        )
        test = dataset_type(
            root=str(config.data_root), train=False, download=config.download, transform=transform
        )
    combined = ConcatDataset([train, test])
    labels = np.concatenate((np.asarray(train.targets), np.asarray(test.targets)))
    return combined, labels


def _digest(train: Iterable[Iterable[int]], test: Iterable[Iterable[int]]) -> str:
    payload = {
        "train": [list(map(int, values)) for values in train],
        "test": [list(map(int, values)) for values in test],
    }
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _fedssm_compatible_indices(config: Config, labels: np.ndarray) -> list[list[int]]:
    indices = np.arange(len(labels))
    result: list[list[int]] = []
    if config.split == "dirichlet":
        by_label = {label: indices[labels == label] for label in np.unique(labels)}
        proportions = np.random.dirichlet(
            config.dirichlet_alpha * np.ones(config.num_clients), size=len(by_label)
        )
        for client_id in range(config.num_clients):
            selected: list[int] = []
            for label_id, class_indices in enumerate(by_label.values()):
                count = int(len(class_indices) * proportions[label_id][client_id])
                # Keep the zero-count call because the historical loader advances RNG.
                chosen = np.random.choice(class_indices, count, replace=False)
                selected.extend(int(value) for value in chosen)
            result.append(selected)
    else:
        shards = np.array_split(np.argsort(labels), config.num_shards)
        per_client = config.num_shards // config.num_clients
        for _ in range(config.num_clients):
            selected = []
            for shard_id in np.random.choice(config.num_shards, per_client, replace=False):
                selected.extend(int(value) for value in shards[int(shard_id)])
            result.append(selected)
    return result


def _disjoint_indices(config: Config, labels: np.ndarray) -> list[list[int]]:
    result: list[list[int]] = [[] for _ in range(config.num_clients)]
    if config.split == "dirichlet":
        for label in np.unique(labels):
            class_indices = np.flatnonzero(labels == label)
            np.random.shuffle(class_indices)
            proportions = np.random.dirichlet(
                config.dirichlet_alpha * np.ones(config.num_clients)
            )
            cuts = (np.cumsum(proportions)[:-1] * len(class_indices)).astype(int)
            for client_id, chunk in enumerate(np.split(class_indices, cuts)):
                result[client_id].extend(int(value) for value in chunk)
    else:
        sorted_indices = np.argsort(labels)
        shards = list(np.array_split(sorted_indices, config.num_shards))
        order = np.random.permutation(config.num_shards)
        per_client = config.num_shards // config.num_clients
        for client_id in range(config.num_clients):
            assigned = order[client_id * per_client : (client_id + 1) * per_client]
            for shard_id in assigned:
                result[client_id].extend(int(value) for value in shards[int(shard_id)])
    return result


def make_partition(config: Config, labels: np.ndarray) -> Partition:
    selected = (
        _fedssm_compatible_indices(config, labels)
        if config.partition_mode == "fedssm-compatible"
        else _disjoint_indices(config, labels)
    )
    frozen_train: list[tuple[int, ...]] = []
    frozen_test: list[tuple[int, ...]] = []
    for values in selected:
        if len(values) < 2:
            raise RuntimeError("a client received fewer than two samples")
        local = Subset(range(len(labels)), values)
        train_size = int((1.0 - config.test_ratio) * len(local))
        test_size = len(local) - train_size
        train_split, test_split = random_split(local, [train_size, test_size])
        frozen_train.append(tuple(int(values[index]) for index in train_split.indices))
        frozen_test.append(tuple(int(values[index]) for index in test_split.indices))

    flattened = [value for client in selected for value in client]
    unique = set(flattened)
    overlap_count = len(flattened) - len(unique)
    unassigned_count = len(labels) - len(unique)
    return Partition(
        tuple(frozen_train),
        tuple(frozen_test),
        _digest(frozen_train, frozen_test),
        overlap_count,
        unassigned_count,
    )


def make_loaders(
    config: Config, partition: Partition, dataset: Dataset[Any]
) -> tuple[list[DataLoader[Any]], list[DataLoader[Any]]]:
    train = [
        DataLoader(
            Subset(dataset, list(indices)), batch_size=config.batch_size, shuffle=True
        )
        for indices in partition.train_indices
    ]
    test = [
        DataLoader(
            Subset(dataset, list(indices)), batch_size=config.batch_size, shuffle=False
        )
        for indices in partition.test_indices
    ]
    return train, test


def calibration_loader(
    dataset: Dataset[Any], batch_size: int, client_id: int, limit: int
) -> DataLoader[Any]:
    subset = Subset(dataset, list(range(min(limit, len(dataset)))))
    generator = torch.Generator().manual_seed(90_000 + client_id)
    return DataLoader(subset, batch_size=batch_size, shuffle=False, generator=generator)

