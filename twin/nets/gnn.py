"""Family 6: graph neural network (message passing) on the grid graph.

Nodes = cells. Edges = the 8 neighbours, each carrying its offset (dy, dx) as an edge feature.
One round: message m_ij = MLP([h_i, h_j, e_ij]) for each neighbour j, mean over neighbours, node update
h_i += MLP([h_i, mean_m]). k rounds = information travels k cells per step. Implemented with tensor shifts
(no torch_geometric, which the judge sandbox does not have).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _shift(x, dy, dx):
    """Value of neighbour (i+dy, j+dx) at every cell; edges replicate."""
    p = F.pad(x, (1, 1, 1, 1), mode="replicate")
    h, w = x.shape[2], x.shape[3]
    return p[:, :, 1 + dy:1 + dy + h, 1 + dx:1 + dx + w]


class MPLayer(nn.Module):
    """m_ij = W2 . gelu(A h_i + B h_j + E e_ij), mean over j.  Same function as MLP([h_i, h_j, e_ij]) with one
    hidden layer, computed cheaply: A h and B h once per layer (not per edge), and W2 applied after the mean
    (it is linear, so mean(W2 x) = W2 mean(x))."""

    def __init__(self, d: int):
        super().__init__()
        self.a = nn.Conv2d(d, d, 1)
        self.b = nn.Conv2d(d, d, 1, bias=False)
        self.e = nn.Parameter(torch.randn(len(OFFSETS), d) * 0.1)        # = E @ (dy, dx) per offset, learned
        self.w2 = nn.Conv2d(d, d, 1)
        self.upd = nn.Sequential(nn.Conv2d(2 * d, d, 1), nn.GELU(), nn.Conv2d(d, d, 1))

    def forward(self, h):
        ah, bh = self.a(h), self.b(h)
        agg = 0
        for k, (dy, dx) in enumerate(OFFSETS):
            agg = agg + F.gelu(ah + _shift(bh, dy, dx) + self.e[k].view(1, -1, 1, 1))
        return h + self.upd(torch.cat([h, self.w2(agg / len(OFFSETS))], 1))


class GridGNN(nn.Module):
    def __init__(self, n_in: int, n_out: int, width: int = 64, rounds: int = 6):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(n_in, width, 1), nn.GELU(), nn.Conv2d(width, width, 1))
        self.layers = nn.Sequential(*[MPLayer(width) for _ in range(rounds)])
        self.dec = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, n_out, 1))
        nn.init.zeros_(self.dec[-1].weight); nn.init.zeros_(self.dec[-1].bias)

    def forward(self, x):
        return self.dec(self.layers(self.enc(x)))
