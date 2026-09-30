"""Small end-to-end test that exercises warm-up and DSA routing."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from dsapfl.config import Config
from dsapfl.data import make_partition
from dsapfl.trainer import run_method, seed_everything


class TinyNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False)
        self.fc = nn.Linear(4, 2)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value = torch.relu(self.conv(inputs)).mean(dim=(2, 3))
        return self.fc(value), value


class TrainerIntegrationTests(unittest.TestCase):
    def test_two_round_dsapfl_run(self) -> None:
        generator = torch.Generator().manual_seed(3)
        inputs = torch.randn(40, 3, 8, 8, generator=generator)
        labels_tensor = torch.tensor([0] * 20 + [1] * 20)
        dataset = TensorDataset(inputs, labels_tensor)
        labels = labels_tensor.numpy()

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            config = Config(
                output_dir=output,
                data_root=output,
                dataset="cifar10",
                method="dsapfl",
                rounds=2,
                num_clients=2,
                batch_size=4,
                split="pathological",
                num_shards=4,
                partition_mode="disjoint",
                device="cpu",
                warmup_rounds=1,
                calibration_limit=8,
                save_checkpoint=False,
            )
            seed_everything(config)
            partition = make_partition(config, np.asarray(labels))
            with patch(
                "dsapfl.trainer.build_model", side_effect=lambda _dataset: TinyNetwork()
            ):
                summary = run_method(
                    "dsapfl", config, dataset, labels, partition, output
                )

            self.assertEqual(summary["rounds"], 2)
            self.assertTrue((output / "results.csv").is_file())
            self.assertTrue((output / "client_results.csv").is_file())
            self.assertTrue((output / "conceptor_stats.csv").is_file())
            self.assertLessEqual(summary["mean_proposal_retention"], 1.000001)


if __name__ == "__main__":
    unittest.main()
