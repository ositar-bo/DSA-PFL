"""Determinism and disjointness tests for client partitioning."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from dsapfl.config import Config
from dsapfl.data import make_partition


class PartitionTests(unittest.TestCase):
    def _config(self, mode: str) -> Config:
        root = Path(tempfile.gettempdir())
        return Config(
            output_dir=root / "dsapfl-test-output",
            data_root=root,
            num_clients=4,
            partition_mode=mode,
            dirichlet_alpha=0.5,
        )

    def test_disjoint_partition_assigns_every_sample_once(self) -> None:
        config = self._config("disjoint")
        labels = np.repeat(np.arange(4), 100)
        np.random.seed(config.partition_seed)
        torch.manual_seed(config.torch_seed)
        partition = make_partition(config, labels)
        self.assertEqual(partition.overlap_count, 0)
        self.assertEqual(partition.unassigned_count, 0)

    def test_partition_replay_is_deterministic(self) -> None:
        config = self._config("fedssm-compatible")
        labels = np.repeat(np.arange(4), 100)
        digests = []
        for _ in range(2):
            np.random.seed(config.partition_seed)
            torch.manual_seed(config.torch_seed)
            digests.append(make_partition(config, labels).digest)
        self.assertEqual(digests[0], digests[1])


if __name__ == "__main__":
    unittest.main()
