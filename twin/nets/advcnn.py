"""Family 11 (v2, spec V2-E7): one advect-and-correct model (PARADIS-style operator split inside every step).

One forecast day = 2 substeps. The model carries two states through the day:
  z  [6]   the anomaly itself (starts as the last input day)
  h  [64]  CNN hidden features (starts as inp(x), the same 3x3 input conv as the v1 CNN `r3_cnn`)
Each substep s:
  1. velocity: a small CNN on h predicts a displacement (dy, dx) per feature group: one group per anomaly variable
     (6) + `lat_groups` groups of learned channels. The vector length is squashed to < max_disp / 2 cells per
     substep, so the TOTAL displacement per day is < max_disp = 3 cells (spec LOC-1 margin, V2-E7).
  2. transport: u = [z, L(h)] (6 + `lat` = 32 channels; L = 1x1 lift) is advected semi-Lagrangian (grid_sample,
     bicubic by default, border padding, align_corners=True) and mixed with the unadvected u through a learned
     per-channel weight a = sigmoid(alpha):  u' = u + a * (adv(u) - u).  z <- z + a_z * (adv(z) - z);
     h <- h + P(u' - u) (P = 1x1 projection back, no bias).
  3. correction: residual blocks on h (local CNN; the v1 CNN's blocks: substep 1 gets block 0, substep 2 blocks 1, 2).
Output: [z_end - z_last + out(h)[:6], out(h)[6]] = change of the anomaly (6 maps) + the rain-gate logit (spec RAIN-3).

With zero velocity adv(u) = u, so the net is EXACTLY the v1 CNN (`inp` -> 3 blocks -> `out`); the velocity heads start
at zero output, so a warm start from r3_cnn (train_nn --warm-start) begins at r3_cnn's forecast and only has to learn
the transport. Gradients reach the velocity because a = 0.5 and P is random at the start.

E7b (spec V2-E7b): `block_k=[3, 1, 1]` turns blocks 1-2 into pointwise (1x1) "reaction" blocks, `interp="bilinear"`,
`max_disp=2.5`, so `reach_bound()` (theoretical day-1 reach, Euclidean cells) is 9.57 <= 10 instead of 18.6.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class ResBlock(nn.Module):
    """Same as cnn.ResBlock (same parameter names), with a kernel size: 3 (spatial) or 1 (pointwise reaction)."""

    def __init__(self, ch: int, k: int = 3):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, k, padding=k // 2, padding_mode="replicate")
        self.c2 = nn.Conv2d(ch, ch, k, padding=k // 2, padding_mode="replicate")
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.c2(self.act(self.c1(self.act(x))))


def _c3(ci, co):
    return nn.Conv2d(ci, co, 3, padding=1, padding_mode="replicate")


def advect(field: torch.Tensor, disp: torch.Tensor, mode: str = "bicubic") -> torch.Tensor:
    """Semi-Lagrangian step. field [B,G,C,H,W] (G groups of C channels), disp [B,G,2,H,W] = (dy, dx) in cells that
    the air moves during the substep -> field read at the departure points x - disp (border padding)."""
    b, g, c, h, w = field.shape
    yy, xx = torch.meshgrid(torch.arange(h, device=field.device, dtype=field.dtype),
                            torch.arange(w, device=field.device, dtype=field.dtype), indexing="ij")
    dep_y = yy - disp[:, :, 0]; dep_x = xx - disp[:, :, 1]                           # [B,G,H,W]
    grid = torch.stack([2 * dep_x / (w - 1) - 1, 2 * dep_y / (h - 1) - 1], dim=-1)     # x first for grid_sample
    out = F.grid_sample(field.reshape(b * g, c, h, w), grid.reshape(b * g, h, w, 2),
                        mode=mode, padding_mode="border", align_corners=True)
    return out.reshape(b, g, c, h, w)


def squash(r: torch.Tensor, cap: float) -> torch.Tensor:
    """r [B,G,2,H,W] raw -> displacement with |d| < cap (radial tanh, smooth at 0, d ~ cap * r for small r)."""
    n = torch.sqrt((r ** 2).sum(dim=2, keepdim=True) + 1e-6)
    return r * (cap * torch.tanh(n) / n)


class AdvectCorrectNet(nn.Module):
    def __init__(self, n_in: int, n_out: int, width: int = 64, blocks: int = 3, split: int = 1, lat: int = 26,
                 lat_groups: int = 2, flow_width: int = 32, max_disp: float = 3.0, substeps: int = 2,
                 interp: str = "bicubic", block_k=None):
        super().__init__()
        assert lat % lat_groups == 0 and 0 < split < blocks and substeps == 2
        self.n_var, self.lat, self.lat_groups, self.interp = 6, lat, lat_groups, interp
        self.block_k = list(block_k) if block_k is not None else [3] * blocks
        assert len(self.block_k) == blocks and all(k in (1, 3) for k in self.block_k)
        self.max_disp = max_disp
        self.cap = max_disp / substeps                                   # per-substep cap -> total < max_disp
        self.n_groups = self.n_var + lat_groups
        self.inp = _c3(n_in, width)                                      # same names/shapes as LocalCNN (warm start)
        self.body = nn.Sequential(*[ResBlock(width, k) for k in self.block_k])
        self.out = nn.Conv2d(width, n_out, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        self.split = [(0, split), (split, blocks)]                       # blocks used by substep 1 and 2
        self.flow = nn.ModuleList()
        self.lift = nn.ModuleList()
        self.proj = nn.ModuleList()
        for _ in range(substeps):
            f = nn.Sequential(_c3(width, flow_width), nn.GELU(), nn.Conv2d(flow_width, 2 * self.n_groups, 1))
            nn.init.zeros_(f[-1].weight); nn.init.zeros_(f[-1].bias)    # start with no motion
            self.flow.append(f)
            self.lift.append(nn.Conv2d(width, lat, 1))
            self.proj.append(nn.Conv2d(self.n_var + lat, width, 1, bias=False))
        self.alpha = nn.Parameter(torch.zeros(substeps, self.n_var + lat))   # a = sigmoid(alpha) = 0.5 at start
        self.last_disp = None                                            # [B,G,2,H,W] per-day total (diagnostics)

    def reach_bound(self) -> float:
        """Theoretical day-1 reach in Euclidean cells (spec V2-E7b): sqrt 2 x (sum of stencil half-sides on the longest
        path: input 3x3 conv 1, each lookup 1 (bilinear) or 2 (bicubic), each 3x3 conv in the blocks 1) + displacement cap.
        The velocity-head 3x3 stencil (half-side 1) is never longer than the lookup it steers, so it adds nothing."""
        look = {"bilinear": 1, "bicubic": 2}[self.interp]
        half = 1 + look * len(self.split) + sum(2 for k in self.block_k if k == 3)
        return math.sqrt(2) * half + self.max_disp

    def forward(self, x):
        b, _, hh, ww = x.shape
        z0 = x[:, 12:18]                                                 # last input day (make_input order)
        z = z0
        h = self.inp(x)
        total = 0.0
        for s, (i0, i1) in enumerate(self.split):
            d = squash(self.flow[s](h).reshape(b, self.n_groups, 2, hh, ww), self.cap)
            total = total + d
            lt = self.lift[s](h)
            zu = advect(z[:, :, None], d[:, :self.n_var], self.interp)[:, :, 0]
            lu = advect(lt.reshape(b, self.lat_groups, -1, hh, ww), d[:, self.n_var:], self.interp).reshape(b, -1, hh, ww)
            a = torch.sigmoid(self.alpha[s])[None, :, None, None]
            du = a * (torch.cat([zu, lu], 1) - torch.cat([z, lt], 1))   # mixed with the unadvected path
            z = z + du[:, :self.n_var]
            h = h + self.proj[s](du)
            for k in range(i0, i1):                                      # local correction on the advected features
                h = self.body[k](h)
        self.last_disp = total.detach() if torch.is_tensor(total) else None
        o = self.out(h)
        return torch.cat([z - z0 + o[:, :self.n_var], o[:, self.n_var:]], 1)


def warm_start(net: nn.Module, state_dict: dict) -> tuple[list[str], list[str]]:
    """Copy every tensor whose name AND shape match (e.g. r3_cnn's inp/body/out into AdvectCorrectNet). A 3x3 conv
    weight going into a 1x1 conv of the same name (E7b reaction blocks) is summed over its taps (same response to a
    smooth field). Returns (reused names, new-model names left at their initialisation)."""
    own = net.state_dict()
    reused = [k for k, v in state_dict.items() if k in own and own[k].shape == v.shape]
    own.update({k: state_dict[k] for k in reused})
    for k, v in state_dict.items():
        if k in own and k not in reused and v.dim() == 4 and own[k].shape == v.shape[:2] + (1, 1):
            own[k] = v.sum(dim=(2, 3), keepdim=True); reused.append(k)
    net.load_state_dict(own)
    return reused, [k for k in own if k not in reused]
