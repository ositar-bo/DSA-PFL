"""Command-line interface for DSA-PFL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import Config
from .routing import Conceptor, dsapfl_fusion
from .trainer import resolve_device, run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dynamic Subspace Adaptation for Personalized Federated Learning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", choices=("cifar10", "cifar100", "fashionmnist", "tinyimagenet"), default="cifar10")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--method", choices=("dsapfl", "fedavg", "both"), default="dsapfl")
    parser.add_argument("--split", choices=("dirichlet", "pathological"), default="dirichlet")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--num-shards", type=int, default=50)
    parser.add_argument("--partition-mode", choices=("fedssm-compatible", "disjoint"), default="fedssm-compatible")
    parser.add_argument("--aggregation", choices=("uniform", "sample"), default="uniform")
    parser.add_argument("--initialization", choices=("independent", "shared"), default="independent")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--torch-seed", type=int, default=543)
    parser.add_argument("--partition-seed", type=int, default=500)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--warmup-rounds", type=int, default=5)
    parser.add_argument("--aperture", type=float, default=10.0)
    parser.add_argument("--tau", type=float, default=0.25)
    parser.add_argument("--lambda-sub", type=float, default=1.0)
    parser.add_argument("--calibration-limit", type=int, default=256)
    parser.add_argument("--conceptor-solver", choices=("direct", "auto"), default="direct")
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def self_test() -> None:
    local = {"layer.weight": torch.zeros(3, 2), "bn.running_mean": torch.zeros(3)}
    global_state = {"layer.weight": torch.ones(3, 2), "bn.running_mean": torch.ones(3)}
    conceptor = Conceptor("layer", 2, 3, 8, torch.diag(torch.tensor([0.0, 0.5, 1.0])))
    fused, retention = dsapfl_fusion(
        global_state, local, {"layer": conceptor}, {"layer": 0.0},
        {"layer.weight": "layer"}, "cpu", tau=0.25, lambda_sub=1.0,
    )
    expected = torch.tensor([[1.0, 1.0], [2 / 3, 2 / 3], [0.5, 0.5]])
    if not torch.allclose(fused["layer.weight"], expected, atol=1e-6):
        raise AssertionError("spectral routing self-test failed")
    if not 0.0 < retention <= 1.0:
        raise AssertionError("routing must be non-expansive")
    if not torch.equal(fused["bn.running_mean"], global_state["bn.running_mean"]):
        raise AssertionError("non-routed state must follow the global aggregate")
    print("SELF_TEST_OK " + json.dumps({"proposal_retention": retention}))


def main() -> None:
    args = build_parser().parse_args()
    if args.self_test:
        self_test()
        return
    split_tag = f"dir{args.alpha:g}" if args.split == "dirichlet" else f"path{args.num_shards}"
    output = args.output_dir or Path("runs") / f"{args.dataset}_{split_tag}_{args.method}_{args.rounds}r"
    config = Config(
        output_dir=output,
        data_root=args.data_root,
        dataset=args.dataset,
        download=args.download,
        method=args.method,
        rounds=args.rounds,
        num_clients=args.num_clients,
        local_epochs=args.local_epochs,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        batch_size=args.batch_size,
        test_ratio=args.test_ratio,
        split=args.split,
        dirichlet_alpha=args.alpha,
        num_shards=args.num_shards,
        partition_mode=args.partition_mode,
        aggregation=args.aggregation,
        initialization=args.initialization,
        torch_seed=args.torch_seed,
        partition_seed=args.partition_seed,
        device=resolve_device(args.device),
        warmup_rounds=args.warmup_rounds,
        aperture=args.aperture,
        tau=args.tau,
        lambda_sub=args.lambda_sub,
        calibration_limit=args.calibration_limit,
        conceptor_solver=args.conceptor_solver,
        save_checkpoint=not args.no_checkpoint,
    )
    run_experiment(config)


if __name__ == "__main__":
    main()

