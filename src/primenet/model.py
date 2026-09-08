"""Tiny models: an MLP and a 1D-CNN variant. Both are trivially CPU-sized (<200k params)."""

from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: tuple[int, ...] = (128, 128, 64)):
        super().__init__()
        dims = [in_dim, *hidden]
        layers: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.GELU()]
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class PrimeCNN(nn.Module):
    """1D convolution over the feature axis: local patterns across bits/residues."""

    def __init__(self, in_dim: int, channels: tuple[int, ...] = (32, 64), kernel: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        prev = 1
        for c in channels:
            layers += [nn.Conv1d(prev, c, kernel, padding=kernel // 2), nn.GELU()]
            prev = c
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prev * in_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x.unsqueeze(1))).squeeze(-1)


def build_model(name: str, in_dim: int, hidden: tuple[int, ...] = (128, 128, 64)) -> nn.Module:
    if name == "mlp":
        return MLP(in_dim, hidden)
    if name == "cnn":
        return PrimeCNN(in_dim)
    raise ValueError(f"unknown model {name!r}; choose 'mlp' or 'cnn'")
