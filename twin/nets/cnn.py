"""Family 3: local ResNet CNN at full resolution. Receptive field radius = 1 + 2 * blocks cells per step."""
from __future__ import annotations

import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, padding_mode="replicate")
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, padding_mode="replicate")
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.c2(self.act(self.c1(self.act(x))))


class LocalCNN(nn.Module):
    def __init__(self, n_in: int, n_out: int, width: int = 64, blocks: int = 3):
        super().__init__()
        self.inp = nn.Conv2d(n_in, width, 3, padding=1, padding_mode="replicate")
        self.body = nn.Sequential(*[ResBlock(width) for _ in range(blocks)])
        self.out = nn.Conv2d(width, n_out, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)   # starts as "anomaly persists"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.body(self.inp(x)))
