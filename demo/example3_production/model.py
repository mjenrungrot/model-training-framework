"""
Example 3 model: small MLP.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn


class SmallMLP(nn.Module):
    def __init__(
        self,
        input_size: int = 64,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_features = input_size
        for _ in range(num_layers):
            layers += [nn.Linear(in_features, hidden_size), nn.ReLU()]
            in_features = hidden_size
        layers.append(nn.Linear(in_features, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast("torch.Tensor", self.net(x))
