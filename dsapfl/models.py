"""ResNet-18 variants used by the DSA-PFL experiments."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResidualBlock(nn.Module):
    """The basic block used by the original CIFAR-10 implementation."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.relu(self.left(inputs) + self.shortcut(inputs))


class Cifar10ResNet18(nn.Module):
    """Exact small-image ResNet-18 used in the completed CIFAR-10 runs."""

    def __init__(self) -> None:
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self._make_layer(64, 2, 1)
        self.layer2 = self._make_layer(128, 2, 2)
        self.layer3 = self._make_layer(256, 2, 2)
        self.layer4 = self._make_layer(512, 2, 2)
        self.fc = nn.Linear(512, 10)

    def _make_layer(self, channels: int, blocks: int, stride: int) -> nn.Sequential:
        result = []
        for block_stride in [stride] + [1] * (blocks - 1):
            result.append(ResidualBlock(self.in_channels, channels, block_stride))
            self.in_channels = channels
        return nn.Sequential(*result)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value = self.layer4(self.layer3(self.layer2(self.layer1(self.conv1(inputs)))))
        features = F.avg_pool2d(value, 4).reshape(value.size(0), -1)
        return self.fc(features), features


class WrappedResNet18(nn.Module):
    """Torchvision ResNet-18 with the logits/features interface used by the runner."""

    def __init__(
        self,
        num_classes: int,
        small_image: bool = False,
        legacy_public_forward: bool = False,
    ) -> None:
        super().__init__()
        try:
            network = models.resnet18(weights=None, num_classes=num_classes)
        except TypeError:  # torchvision < 0.13
            network = models.resnet18(pretrained=False, num_classes=num_classes)
        if small_image:
            network.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
            network.maxpool = nn.Identity()
        self.resnet = network
        self.legacy_public_forward = legacy_public_forward

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        net = self.resnet
        if self.legacy_public_forward:
            # The original public CIFAR-100/FashionMNIST wrappers feed conv1
            # directly into layer1, leaving bn1/relu/maxpool unused. Preserve
            # this unusual behavior because the released results depend on it.
            value = net.layer1(net.conv1(inputs))
        else:
            value = net.layer1(net.maxpool(net.relu(net.bn1(net.conv1(inputs)))))
        value = net.layer4(net.layer3(net.layer2(value)))
        features = torch.flatten(net.avgpool(value), 1)
        return net.fc(features), features


def build_model(dataset: str) -> nn.Module:
    if dataset == "cifar10":
        return Cifar10ResNet18()
    if dataset == "cifar100":
        # Preserves the original FedSSM public-code architecture.
        return WrappedResNet18(
            num_classes=100, small_image=False, legacy_public_forward=True
        )
    if dataset == "fashionmnist":
        return WrappedResNet18(
            num_classes=10, small_image=False, legacy_public_forward=True
        )
    if dataset == "tinyimagenet":
        return WrappedResNet18(num_classes=200, small_image=True)
    raise ValueError(f"unsupported dataset: {dataset}")
