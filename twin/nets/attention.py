"""Families 7a/7b: transformers with LOCAL window attention (Swin style), so a shock cannot jump across India in one step.

hybrid : CNN stem at full resolution -> window-attention blocks -> CNN head   (CNN + transformer)
vit    : 2x2 patch embedding -> window-attention blocks on the 16x18 token grid -> un-patch     (pure transformer)

Window attention: tokens attend only inside non-overlapping w x w windows; every second block shifts the
windows by w/2 (Swin), so information crosses window borders gradually, like a conv with a bigger kernel.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class WindowBlock(nn.Module):
    def __init__(self, dim: int, heads: int, window: int, shift: int):
        super().__init__()
        self.w, self.shift = window, shift
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)      # per-token norm (not across the grid)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.pos = nn.Parameter(torch.zeros(1, window * window, dim))
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, x):                                    # x [B, C, H, W]
        b, c, h, w = x.shape
        ws = self.w
        ph, pw = (-h) % ws, (-w) % ws
        y = F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x
        if self.shift:
            y = torch.roll(y, (-self.shift, -self.shift), (2, 3))
        H, W = y.shape[2], y.shape[3]
        t = y.reshape(b, c, H // ws, ws, W // ws, ws).permute(0, 2, 4, 3, 5, 1).reshape(-1, ws * ws, c)
        q = self.n1(t) + self.pos
        t = t + self.attn(q, q, q, need_weights=False)[0]
        t = t + self.mlp(self.n2(t))
        y = t.reshape(b, H // ws, W // ws, ws, ws, c).permute(0, 5, 1, 3, 2, 4).reshape(b, c, H, W)
        if self.shift:
            y = torch.roll(y, (self.shift, self.shift), (2, 3))
        return y[:, :, :h, :w]


def _c(ci, co):
    return nn.Conv2d(ci, co, 3, padding=1, padding_mode="replicate")


class HybridCNNTransformer(nn.Module):
    def __init__(self, n_in: int, n_out: int, width: int = 64, heads: int = 4, window: int = 4, blocks: int = 2):
        super().__init__()
        self.stem = nn.Sequential(_c(n_in, width), nn.GELU(), _c(width, width), nn.GELU())
        self.blocks = nn.Sequential(*[WindowBlock(width, heads, window, (window // 2) * (i % 2)) for i in range(blocks)])
        self.head = nn.Sequential(_c(width, width), nn.GELU(), nn.Conv2d(width, n_out, 1))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


class WindowViT(nn.Module):
    def __init__(self, n_in: int, n_out: int, width: int = 96, heads: int = 4, window: int = 4, blocks: int = 2, patch: int = 2):
        super().__init__()
        self.p = patch
        self.embed = nn.Conv2d(n_in, width, patch, stride=patch)
        self.blocks = nn.Sequential(*[WindowBlock(width, heads, window, (window // 2) * (i % 2)) for i in range(blocks)])
        self.out = nn.Linear(width, n_out * patch * patch)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        self.n_out = n_out

    def forward(self, x):
        b, _, h, w = x.shape
        t = self.blocks(self.embed(x))                       # [B, C, h/p, w/p]
        y = self.out(t.permute(0, 2, 3, 1))                  # [B, h/p, w/p, n_out*p*p]
        y = y.reshape(b, h // self.p, w // self.p, self.n_out, self.p, self.p)
        return y.permute(0, 3, 1, 4, 2, 5).reshape(b, self.n_out, h, w)
