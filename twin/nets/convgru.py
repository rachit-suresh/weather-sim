"""Family 5 (time series): CNN + recurrent memory (ConvGRU).

Step 1: the GRU reads the 3 input days one by one (oldest first) to build a hidden state h.
Every later forecast day: h is updated with only the newest day, so memory runs across the whole
rollout (longer than the 3-day window). A small conv head turns h into the change of the anomaly.
Per step receptive field: 2 cells (encoder) + 1 per GRU update + 2 (head).
"""
from __future__ import annotations

import torch
from torch import nn


def _c(ci, co, k=3):
    return nn.Conv2d(ci, co, k, padding=k // 2, padding_mode="replicate")


class ConvGRUCell(nn.Module):
    def __init__(self, ci, ch):
        super().__init__()
        self.gates = _c(ci + ch, 2 * ch)
        self.cand = _c(ci + ch, ch)

    def forward(self, x, h):
        r, u = torch.sigmoid(self.gates(torch.cat([x, h], 1))).chunk(2, 1)
        c = torch.tanh(self.cand(torch.cat([x, r * h], 1)))
        return (1 - u) * h + u * c


class ConvGRUNet(nn.Module):
    stateful = True

    def __init__(self, n_in: int, n_out: int, width: int = 64):
        super().__init__()
        self.n_ctx = n_in - 18                       # clim, static, sincos, edge
        self.enc = nn.Sequential(_c(6 + self.n_ctx, width), nn.GELU(), _c(width, width), nn.GELU())
        self.cell = ConvGRUCell(width, width)
        self.head = nn.Sequential(_c(width, width), nn.GELU(), nn.Conv2d(width, n_out, 1))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)
        self.width = width

    def forward(self, x, h=None):
        ctx = x[:, 18:]
        days = [x[:, 6 * d:6 * d + 6] for d in range(3)]
        if h is None:
            h = torch.zeros(x.shape[0], self.width, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
            for d in days:
                h = self.cell(self.enc(torch.cat([d, ctx], 1)), h)
        else:
            h = self.cell(self.enc(torch.cat([days[-1], ctx], 1)), h)
        return self.head(h), h
