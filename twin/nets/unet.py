"""Family 4: small U-Net with ONE down-sampling level (32x36 -> 16x18), keeping the per-step receptive field
well under the grader's no-teleport radius. Deeper U-Nets see the whole grid in one step: avoid."""
from __future__ import annotations

import torch
from torch import nn


def _conv(ci, co):
    return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1, padding_mode="replicate"), nn.GELU(),
                         nn.Conv2d(co, co, 3, padding=1, padding_mode="replicate"), nn.GELU())


class SmallUNet(nn.Module):
    def __init__(self, n_in: int, n_out: int, width: int = 48):
        super().__init__()
        self.enc = _conv(n_in, width)
        self.down = nn.AvgPool2d(2)
        self.mid = _conv(width, width * 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec = _conv(width * 3, width)
        self.out = nn.Conv2d(width, n_out, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.enc(x)
        m = self.up(self.mid(self.down(e)))
        return self.out(self.dec(torch.cat([e, m], dim=1)))
