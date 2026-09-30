"""Federated training loop for DSA-PFL and a FedAvg sanity baseline."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .config import Config
from .data import Partition, calibration_loader, load_dataset, make_loaders, make_partition
from .models import build_model
from .routing import (
    Conceptor,
    collect_conceptors,
    conceptor_drift,
    dsapfl_fusion,
    maturity,
    parameter_module_map,
)


RESULT_FIELDS = [
    "method", "round", "personalized_mean_accuracy", "personalized_weighted_accuracy",
    "worst_client_accuracy", "client_accuracy_std", "global_mean_accuracy",
    "global_weighted_accuracy", "mean_conceptor_drift", "mean_occupancy",
    "mean_proposal_retention", "round_seconds",
]
CLIENT_FIELDS = [
    "method", "round", "client", "train_samples", "test_samples",
    "personalized_accuracy", "global_accuracy", "mean_conceptor_drift",
    "mean_occupancy", "proposal_retention",
]
CONCEPTOR_FIELDS = [
    "method", "round", "client", "layer", "dimension", "calibration_samples",
    "drift", "maturity", "occupancy", "used_for_routing",
]


def seed_everything(config: Config) -> None:
    random.seed(config.torch_seed)
    np.random.seed(config.partition_seed)
    torch.manual_seed(config.torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.torch_seed)


def resolve_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def state_digest(states: Iterable[Mapping[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for state in states:
        for key, value in state.items():
            tensor = value.detach().cpu().contiguous()
            digest.update(key.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def aggregate_states(
    states: list[Mapping[str, torch.Tensor]], weights: list[int] | None
) -> dict[str, torch.Tensor]:
    if not states:
        raise ValueError("cannot aggregate an empty list")
    result: dict[str, torch.Tensor] = {}
    if weights is None:
        # Exact operation order of the completed experiments: deepcopy the
        # first state, add the rest, then convert/divide.
        result = copy.deepcopy(dict(states[0]))
        for key in result:
            for state in states[1:]:
                result[key] += state[key]
            result[key] = result[key].float() / len(states)
        return result

    coefficients = np.asarray(weights, dtype=np.float64)
    coefficients /= coefficients.sum()
    for key in states[0]:
        first = states[0][key]
        value = torch.zeros_like(first, dtype=torch.float32)
        for coefficient, state in zip(coefficients, states):
            value.add_(state[key].float(), alpha=float(coefficient))
        result[key] = value
    return result


def _write_csv(path: Path, fields: list[str], rows: Iterable[Mapping[str, Any]]) -> None:
    values = list(rows)
    if not values:
        return
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if new:
            writer.writeheader()
        writer.writerows(values)


def _cpu_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def _weighted(values: list[float], weights: list[int]) -> float:
    return float(np.average(np.asarray(values, dtype=np.float64), weights=weights))


class Client:
    def __init__(
        self,
        config: Config,
        client_id: int,
        train_loader: DataLoader[Any],
        test_loader: DataLoader[Any],
        shared_state: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        self.config = config
        self.client_id = client_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.net = build_model(config.dataset).to(config.device)
        if shared_state is not None:
            self.net.load_state_dict(shared_state)
        self.optimizer = optim.SGD(
            self.net.parameters(), lr=config.learning_rate, momentum=config.momentum
        )
        self.criterion = nn.CrossEntropyLoss()
        self.param_modules = parameter_module_map(self.net)
        self.previous_conceptors: dict[str, Conceptor] | None = None
        self.calibration_loader = calibration_loader(
            self.train_loader.dataset,
            config.batch_size,
            client_id,
            config.calibration_limit,
        )

    def train(self) -> dict[str, torch.Tensor]:
        self.net.train()
        for _ in range(self.config.local_epochs):
            for inputs, labels in self.train_loader:
                inputs = inputs.to(self.config.device)
                labels = labels.to(self.config.device)
                self.optimizer.zero_grad()
                logits, _ = self.net(inputs)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()
        return copy.deepcopy(self.net.state_dict())

    def accuracy(self, model: nn.Module) -> float:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                logits, _ = model(inputs.to(self.config.device))
                labels = labels.to(self.config.device)
                correct += int((logits.argmax(dim=1) == labels).sum().item())
                total += int(labels.numel())
        if total == 0:
            raise RuntimeError(f"client {self.client_id} has an empty test set")
        return 100.0 * correct / total


def _build_participants(
    config: Config,
    train_loaders: list[DataLoader[Any]],
    test_loaders: list[DataLoader[Any]],
) -> tuple[list[Client], nn.Module, str]:
    if config.initialization == "shared":
        template = build_model(config.dataset).to(config.device)
        initial = copy.deepcopy(template.state_dict())
        clients = [
            Client(config, i, train_loaders[i], test_loaders[i], initial)
            for i in range(config.num_clients)
        ]
        global_model = build_model(config.dataset).to(config.device)
        global_model.load_state_dict(initial)
    else:
        clients = [
            Client(config, i, train_loaders[i], test_loaders[i])
            for i in range(config.num_clients)
        ]
        global_model = build_model(config.dataset).to(config.device)
    digest = state_digest(
        [client.net.state_dict() for client in clients] + [global_model.state_dict()]
    )
    return clients, global_model, digest


def run_method(
    method: str,
    config: Config,
    dataset: Any,
    labels: np.ndarray,
    expected_partition: Partition,
    output: Path,
) -> dict[str, Any]:
    seed_everything(config)
    # Replaying the partition here is intentional. Besides recovering the same
    # frozen indices, random_split advances the torch RNG before model creation,
    # exactly as in the completed experiment runner.
    partition = make_partition(config, labels)
    if partition.digest != expected_partition.digest:
        raise RuntimeError("partition replay changed between paired methods")
    train_loaders, test_loaders = make_loaders(config, partition, dataset)
    clients, global_model, initialization_hash = _build_participants(
        config, train_loaders, test_loaders
    )
    train_sizes = [len(loader.dataset) for loader in train_loaders]
    test_sizes = [len(loader.dataset) for loader in test_loaders]
    results: list[dict[str, Any]] = []
    best_accuracy = -math.inf
    best_checkpoint: dict[str, Any] | None = None
    start = time.time()

    for round_id in range(1, config.rounds + 1):
        round_start = time.time()
        print(f"\n--- {method} round {round_id}/{config.rounds} ---", flush=True)
        local_states = [client.train() for client in clients]
        personalized = [client.accuracy(client.net) for client in clients]
        weights = train_sizes if config.aggregation == "sample" else None
        global_state = aggregate_states(local_states, weights)
        global_model.load_state_dict(global_state)
        global_accuracy = [client.accuracy(global_model) for client in clients]

        if method == "dsapfl" and float(np.mean(personalized)) > best_accuracy:
            best_accuracy = float(np.mean(personalized))
            best_checkpoint = {
                "method": method,
                "round": round_id,
                "personalized_mean_accuracy": best_accuracy,
                "partition_sha256": partition.digest,
                "initialization_sha256": initialization_hash,
                "global_state": _cpu_state(global_state),
                "client_states": [_cpu_state(state) for state in local_states],
                "config": {**asdict(config), "output_dir": str(config.output_dir), "data_root": str(config.data_root)},
            }

        current: list[dict[str, Conceptor] | None] = [None] * config.num_clients
        drifts: list[dict[str, float]] = [{} for _ in clients]
        if method == "dsapfl" and round_id >= config.warmup_rounds:
            for client_id, client in enumerate(clients):
                bases = collect_conceptors(
                    client.net,
                    client.calibration_loader,
                    config.device,
                    round_id,
                    config.aperture,
                    config.conceptor_solver,
                )
                previous = client.previous_conceptors
                current[client_id] = bases
                drifts[client_id] = {
                    layer: (
                        conceptor_drift(basis, previous[layer])
                        if previous is not None and layer in previous
                        else 0.0
                    )
                    for layer, basis in bases.items()
                }

        client_rows: list[dict[str, Any]] = []
        conceptor_rows: list[dict[str, Any]] = []
        round_drifts: list[float] = []
        round_occupancies: list[float] = []
        round_retentions: list[float] = []
        for client_id, client in enumerate(clients):
            client_drift = 0.0
            client_occupancy = 0.0
            retention = 1.0
            if method == "fedavg" or round_id <= config.warmup_rounds:
                fused = {key: value.detach().clone() for key, value in global_state.items()}
            else:
                bases = current[client_id]
                if bases is None:
                    raise RuntimeError("missing Conceptor state after warm-up")
                fused, retention = dsapfl_fusion(
                    global_state,
                    local_states[client_id],
                    bases,
                    drifts[client_id],
                    client.param_modules,
                    config.device,
                    config.tau,
                    config.lambda_sub,
                )
                client_drift = float(np.mean(list(drifts[client_id].values())))
                client_occupancy = float(np.mean([basis.occupancy for basis in bases.values()]))
                round_drifts.append(client_drift)
                round_occupancies.append(client_occupancy)
                round_retentions.append(retention)
            client.net.load_state_dict(fused)

            bases = current[client_id]
            if method == "dsapfl" and bases is not None:
                for layer, basis in bases.items():
                    drift = drifts[client_id][layer]
                    conceptor_rows.append({
                        "method": method, "round": round_id, "client": client_id,
                        "layer": layer, "dimension": basis.dimension,
                        "calibration_samples": basis.samples, "drift": drift,
                        "maturity": maturity(drift, config.tau),
                        "occupancy": basis.occupancy,
                        "used_for_routing": int(round_id > config.warmup_rounds),
                    })
                client.previous_conceptors = bases

            client_rows.append({
                "method": method, "round": round_id, "client": client_id,
                "train_samples": train_sizes[client_id], "test_samples": test_sizes[client_id],
                "personalized_accuracy": personalized[client_id],
                "global_accuracy": global_accuracy[client_id],
                "mean_conceptor_drift": client_drift,
                "mean_occupancy": client_occupancy,
                "proposal_retention": retention,
            })

        row = {
            "method": method,
            "round": round_id,
            "personalized_mean_accuracy": float(np.mean(personalized)),
            "personalized_weighted_accuracy": _weighted(personalized, test_sizes),
            "worst_client_accuracy": float(np.min(personalized)),
            "client_accuracy_std": float(np.std(personalized)),
            "global_mean_accuracy": float(np.mean(global_accuracy)),
            "global_weighted_accuracy": _weighted(global_accuracy, test_sizes),
            "mean_conceptor_drift": float(np.mean(round_drifts)) if round_drifts else 0.0,
            "mean_occupancy": float(np.mean(round_occupancies)) if round_occupancies else 0.0,
            "mean_proposal_retention": float(np.mean(round_retentions)) if round_retentions else 1.0,
            "round_seconds": time.time() - round_start,
        }
        results.append(row)
        _write_csv(output / "results.csv", RESULT_FIELDS, [row])
        _write_csv(output / "client_results.csv", CLIENT_FIELDS, client_rows)
        _write_csv(output / "conceptor_stats.csv", CONCEPTOR_FIELDS, conceptor_rows)
        print(
            f"PAcc={row['personalized_mean_accuracy']:.6f}% "
            f"worst={row['worst_client_accuracy']:.6f}% "
            f"global={row['global_mean_accuracy']:.6f}% "
            f"drift={row['mean_conceptor_drift']:.6f} "
            f"retention={row['mean_proposal_retention']:.6f} "
            f"seconds={row['round_seconds']:.1f}",
            flush=True,
        )

    if method == "dsapfl" and config.save_checkpoint:
        if best_checkpoint is None:
            raise RuntimeError("best DSA-PFL checkpoint was not populated")
        torch.save(best_checkpoint, output / "best_dsapfl.pt")
    values = np.asarray([row["personalized_mean_accuracy"] for row in results])
    return {
        "method": method,
        "rounds": len(results),
        "final_personalized_accuracy": float(values[-1]),
        "best_personalized_accuracy": float(values.max()),
        "best_round": int(values.argmax() + 1),
        "last10_mean": float(values[-10:].mean()),
        "last10_std": float(values[-10:].std()),
        "final_worst_client_accuracy": float(results[-1]["worst_client_accuracy"]),
        "final_client_accuracy_std": float(results[-1]["client_accuracy_std"]),
        "mean_conceptor_drift": float(np.mean([row["mean_conceptor_drift"] for row in results[config.warmup_rounds:]])) if len(results) > config.warmup_rounds else 0.0,
        "mean_occupancy": float(np.mean([row["mean_occupancy"] for row in results[config.warmup_rounds:]])) if len(results) > config.warmup_rounds else 0.0,
        "mean_proposal_retention": float(np.mean([row["mean_proposal_retention"] for row in results[config.warmup_rounds:]])) if len(results) > config.warmup_rounds else 1.0,
        "runtime_seconds": time.time() - start,
        "partition_sha256": partition.digest,
        "initialization_sha256": initialization_hash,
    }


def run_experiment(config: Config) -> dict[str, Any]:
    config.validate()
    output = config.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "results.csv").exists():
        raise RuntimeError(f"output already contains results: {output}")

    seed_everything(config)
    dataset, labels = load_dataset(config)
    partition = make_partition(config, labels)
    methods = ["fedavg", "dsapfl"] if config.method == "both" else [config.method]
    serializable_config = {
        **asdict(config),
        "output_dir": str(config.output_dir),
        "data_root": str(config.data_root),
        "device": config.device,
    }
    metadata = {
        "method_name": "DSA-PFL",
        "former_name": "DSR-PFL",
        "config": serializable_config,
        "partition_sha256": partition.digest,
        "partition_overlap_count": partition.overlap_count,
        "partition_unassigned_count": partition.unassigned_count,
        "protocol_warning": (
            "fedssm-compatible can overlap clients and leave samples unassigned"
            if config.partition_mode == "fedssm-compatible" else None
        ),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summaries = {
        method: run_method(method, config, dataset, labels, partition, output)
        for method in methods
    }
    if len(summaries) == 2:
        hashes = {value["initialization_sha256"] for value in summaries.values()}
        if len(hashes) != 1:
            raise RuntimeError("paired methods did not share the initialization sequence")
    payload = {"metadata": metadata, "methods": summaries}
    (output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("EXPERIMENT_COMPLETE", flush=True)
    return payload
