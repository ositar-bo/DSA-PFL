"""Dynamic subspace adaptation primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


EPSILON = 1e-12


@dataclass
class Conceptor:
    module: str
    source_round: int
    dimension: int
    samples: int
    matrix: torch.Tensor

    @property
    def occupancy(self) -> float:
        return float(torch.trace(self.matrix.double()).item() / self.dimension)


def routed_modules(model: nn.Module) -> dict[str, nn.Module]:
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    }


def parameter_module_map(model: nn.Module) -> dict[str, str]:
    result: dict[str, str] = {}
    for module_name, module in routed_modules(model).items():
        for parameter_name, _ in module.named_parameters(recurse=False):
            result[f"{module_name}.{parameter_name}"] = module_name
    return result


def _conceptor_matrix(
    features: torch.Tensor, aperture: float, solver: str
) -> torch.Tensor:
    samples = features.shape[1]
    ridge = aperture ** -2
    if solver == "auto" and features.shape[0] > samples:
        gram = features.transpose(0, 1) @ features
        identity = torch.eye(samples, device=features.device, dtype=features.dtype)
        solved = torch.linalg.solve(gram + (samples * ridge) * identity, features.transpose(0, 1))
        matrix = features @ solved
    else:
        covariance = (features @ features.transpose(0, 1)) / samples
        identity = torch.eye(
            covariance.shape[0], device=features.device, dtype=features.dtype
        )
        matrix = torch.linalg.solve(covariance + ridge * identity, covariance)
    return 0.5 * (matrix + matrix.transpose(0, 1))


def collect_conceptors(
    model: nn.Module,
    loader: DataLoader[Any],
    device: str,
    source_round: int,
    aperture: float,
    solver: str = "direct",
) -> dict[str, Conceptor]:
    modules = routed_modules(model)
    collected: dict[str, list[torch.Tensor]] = {name: [] for name in modules}
    handles: list[Any] = []

    def make_hook(name: str):
        def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            activation = output.detach()
            if activation.ndim == 4:
                activation = activation.mean(dim=(2, 3))
            elif activation.ndim != 2:
                raise RuntimeError(
                    f"unsupported activation shape for {name}: {tuple(activation.shape)}"
                )
            collected[name].append(activation.float().cpu())

        return hook

    for name, module in modules.items():
        handles.append(module.register_forward_hook(make_hook(name)))
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for inputs, _ in loader:
                model(inputs.to(device))
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    result: dict[str, Conceptor] = {}
    for name, chunks in collected.items():
        if not chunks:
            raise RuntimeError(f"no calibration activations collected for {name}")
        features = torch.cat(chunks, dim=0).transpose(0, 1).contiguous().to(device)
        matrix = _conceptor_matrix(features, aperture, solver)
        result[name] = Conceptor(
            module=name,
            source_round=source_round,
            dimension=int(features.shape[0]),
            samples=int(features.shape[1]),
            matrix=matrix.detach().cpu(),
        )
    return result


def conceptor_drift(current: Conceptor, previous: Conceptor) -> float:
    if current.dimension != previous.dimension:
        raise ValueError("Conceptor dimensions changed between rounds")
    numerator = torch.linalg.vector_norm(
        current.matrix.double() - previous.matrix.double()
    ).item()
    denominator = torch.linalg.vector_norm(previous.matrix.double()).item()
    return float(numerator / (denominator + EPSILON))


def maturity(drift: float, tau: float) -> float:
    return math.exp(-float(drift) / tau)


def dsapfl_fusion(
    global_state: Mapping[str, torch.Tensor],
    local_state: Mapping[str, torch.Tensor],
    conceptors: Mapping[str, Conceptor],
    drifts: Mapping[str, float],
    param_modules: Mapping[str, str],
    device: str,
    tau: float,
    lambda_sub: float,
) -> tuple[dict[str, torch.Tensor], float]:
    """Route a global proposal and return the fused state and norm retention."""

    fused: dict[str, torch.Tensor] = {}
    systems: dict[str, torch.Tensor] = {}
    proposal_sq = 0.0
    routed_sq = 0.0
    for key, global_raw in global_state.items():
        module_name = param_modules.get(key)
        if module_name is None or module_name not in conceptors:
            fused[key] = global_raw.detach().clone()
            continue
        global_value = global_raw.float()
        local_value = local_state[key].float()
        proposal = global_value - local_value
        proposal_matrix = proposal.reshape(proposal.shape[0], -1)
        if module_name not in systems:
            conceptor = conceptors[module_name].matrix.to(
                device=device, dtype=proposal_matrix.dtype
            )
            identity = torch.eye(
                conceptor.shape[0], device=device, dtype=proposal_matrix.dtype
            )
            systems[module_name] = (
                identity
                + lambda_sub * maturity(drifts[module_name], tau) * conceptor
            )
        routed = torch.linalg.solve(systems[module_name], proposal_matrix).reshape_as(proposal)
        fused[key] = local_value + routed
        proposal_sq += float(proposal.double().square().sum().item())
        routed_sq += float(routed.double().square().sum().item())
    retention = math.sqrt(routed_sq) / (math.sqrt(proposal_sq) + EPSILON)
    return fused, retention

