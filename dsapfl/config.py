"""Experiment configuration for DSA-PFL."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    output_dir: Path
    data_root: Path
    dataset: str = "cifar10"
    download: bool = False
    method: str = "dsapfl"
    rounds: int = 100
    num_clients: int = 10
    local_epochs: int = 1
    learning_rate: float = 0.001
    momentum: float = 0.9
    batch_size: int = 64
    test_ratio: float = 0.2
    split: str = "dirichlet"
    dirichlet_alpha: float = 0.1
    num_shards: int = 50
    partition_mode: str = "fedssm-compatible"
    merge_official_splits: bool = True
    aggregation: str = "uniform"
    initialization: str = "independent"
    torch_seed: int = 543
    partition_seed: int = 500
    device: str = "auto"
    warmup_rounds: int = 5
    aperture: float = 10.0
    tau: float = 0.25
    lambda_sub: float = 1.0
    calibration_limit: int = 256
    conceptor_solver: str = "direct"
    save_checkpoint: bool = True

    def validate(self) -> None:
        if self.dataset not in {"cifar10", "cifar100", "fashionmnist", "tinyimagenet"}:
            raise ValueError(f"unsupported dataset: {self.dataset}")
        if self.method not in {"dsapfl", "fedavg", "both"}:
            raise ValueError(f"unsupported method: {self.method}")
        if self.split not in {"dirichlet", "pathological"}:
            raise ValueError(f"unsupported split: {self.split}")
        if self.partition_mode not in {"fedssm-compatible", "disjoint"}:
            raise ValueError(f"unsupported partition mode: {self.partition_mode}")
        if self.aggregation not in {"uniform", "sample"}:
            raise ValueError(f"unsupported aggregation: {self.aggregation}")
        if self.initialization not in {"independent", "shared"}:
            raise ValueError(f"unsupported initialization: {self.initialization}")
        if self.conceptor_solver not in {"direct", "auto"}:
            raise ValueError(f"unsupported Conceptor solver: {self.conceptor_solver}")
        if self.rounds < 1 or self.num_clients < 1 or self.local_epochs < 1:
            raise ValueError("rounds, num_clients, and local_epochs must be positive")
        if self.batch_size < 1 or self.calibration_limit < 1:
            raise ValueError("batch_size and calibration_limit must be positive")
        if not 0.0 < self.test_ratio < 1.0:
            raise ValueError("test_ratio must be in (0, 1)")
        if self.split == "dirichlet" and self.dirichlet_alpha <= 0.0:
            raise ValueError("Dirichlet alpha must be positive")
        if self.split == "pathological":
            if self.num_shards < self.num_clients or self.num_shards % self.num_clients:
                raise ValueError("num_shards must be >= num_clients and divisible by it")
        if self.aperture <= 0.0 or self.tau <= 0.0 or self.lambda_sub < 0.0:
            raise ValueError("aperture/tau must be positive and lambda_sub non-negative")
        if self.warmup_rounds < 1:
            raise ValueError("warmup_rounds must be at least one")

