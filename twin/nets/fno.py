"""Family 8: Fourier neural operator (FourCastNet-style spectral mixing).

Included as a CONTROL experiment: every Fourier mode touches every cell, so a shock reaches the whole grid
in one step. We expect it to fail the grader's 'localized' / 'no teleport' checks. Not a submission candidate.
"""
from __future__ import annotations

import torch
from torch import nn


class SpectralConv(nn.Module):
    def __init__(self, ch: int, m1: int, m2: int):
        super().__init__()
        self.m1, self.m2 = m1, m2
        s = 1 / (ch * ch)
        self.w1 = nn.Parameter(s * torch.randn(ch, ch, m1, m2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(s * torch.randn(ch, ch, m1, m2, dtype=torch.cfloat))

    def forward(self, x):
        b, c, h, w = x.shape
        xf = torch.fft.rfft2(x)
        out = torch.zeros(b, c, h, w // 2 + 1, dtype=torch.cfloat, device=x.device)
        out[:, :, :self.m1, :self.m2] = torch.einsum("bixy,ioxy->boxy", xf[:, :, :self.m1, :self.m2], self.w1)
        out[:, :, -self.m1:, :self.m2] = torch.einsum("bixy,ioxy->boxy", xf[:, :, -self.m1:, :self.m2], self.w2)
        return torch.fft.irfft2(out, s=(h, w))


class FNO(nn.Module):
    def __init__(self, n_in: int, n_out: int, width: int = 48, modes: int = 10, layers: int = 4):
        super().__init__()
        self.lift = nn.Conv2d(n_in, width, 1)
        self.spec = nn.ModuleList([SpectralConv(width, modes, modes) for _ in range(layers)])
        self.pw = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.proj = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, n_out, 1))
        nn.init.zeros_(self.proj[-1].weight); nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x):
        h = self.lift(x)
        for s, p in zip(self.spec, self.pw):
            h = nn.functional.gelu(s(h) + p(h))
        return self.proj(h)
