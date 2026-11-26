"""Autoregressive rollout in normalised-anomaly space, used by BOTH training (with gradients) and inference."""
from __future__ import annotations

import torch

from .common import make_input

Z_CLAMP = 12.0     # anomalies beyond 12 std are unphysical: clamp to keep long rollouts stable


def rollout(net, hist: torch.Tensor, clim_targets: torch.Tensor, sincos_targets: torch.Tensor,
            fixed: torch.Tensor, steps: int) -> torch.Tensor:
    """hist [B,3,6,H,W]; clim_targets [B,steps,6,H,W]; sincos_targets [B,steps,2] -> [B,steps,6,H,W] anomalies."""
    outs = []
    for k in range(steps):
        x = make_input(hist, clim_targets[:, k], sincos_targets[:, k], fixed)
        z = torch.clamp(hist[:, -1] + net(x), -Z_CLAMP, Z_CLAMP)
        outs.append(z)
        hist = torch.cat([hist[:, 1:], z[:, None]], dim=1)
    return torch.stack(outs, dim=1)
