"""Autoregressive rollout in normalised-anomaly space, used by BOTH training (with gradients) and inference.

Stateless nets: out = net(x).  Stateful nets (`stateful = True`, e.g. ConvGRU): out, h = net(x, h); the hidden
state starts as None on every call, so each request is still a pure function of its inputs (null-shock test).

Options (dict `cfg`, stored in the checkpoint as "rollout"):
  edge_relax  [f0, f1, f2]  spec EDG-2 (F4): multiply the anomaly of cells 0/1/2 cells from the edge after every step
  rain_head   "two_part"    spec RAIN-3 (D4): net output channel 6 is the logit of "rain > 1 mm/day"
  rain_ref    {mean, std, anom_std}  global mean/std of rain (mm/day) and the rain anomaly std (sqrt space),
                            needed to turn the gate into anomaly units (rain must use the sqrt transform)
mode "train" uses a smooth gate (sigmoid x amount), mode "infer" a hard gate (amount where logit > 0).
"""
from __future__ import annotations

import torch

from ..config import TP
from .common import edge_relax_map, make_input

Z_CLAMP = 12.0     # anomalies beyond 12 std are unphysical: clamp to keep long rollouts stable


def _gate_rain(z, logit, clim_k, ref, mode):
    """Replace the rain anomaly by the gated amount. clim_k [B,6,H,W] = target-day climatology feature."""
    c = torch.sqrt(torch.relu(clim_k[:, TP] * ref["std"] + ref["mean"]))          # sqrt(climatological rain)
    amount = torch.relu(z[:, TP] * ref["anom_std"] + c)                           # sqrt(rain) >= 0
    g = torch.sigmoid(logit) if mode == "train" else (logit > 0).to(z.dtype)
    z_tp = (g * amount - c) / ref["anom_std"]
    return torch.cat([z[:, :TP], z_tp[:, None], z[:, TP + 1:]], dim=1)


def rollout(net, hist: torch.Tensor, clim_targets: torch.Tensor, sincos_targets: torch.Tensor,
            fixed: torch.Tensor, steps: int, cfg: dict | None = None, mode: str = "train", return_logits: bool = False):
    """hist [B,3,6,H,W]; clim_targets [B,steps,6,H,W]; sincos_targets [B,steps,2] -> [B,steps,6,H,W] anomalies
    (and [B,steps,H,W] rain logits if return_logits and the two-part head is on)."""
    cfg = cfg or {}
    stateful = getattr(net, "stateful", False)
    two_part = cfg.get("rain_head") == "two_part"
    relax = None
    if cfg.get("edge_relax"):
        relax = torch.as_tensor(edge_relax_map(cfg["edge_relax"]), dtype=hist.dtype, device=hist.device)
    h = None
    outs, logits = [], []
    for k in range(steps):
        x = make_input(hist, clim_targets[:, k], sincos_targets[:, k], fixed)
        if stateful:
            out, h = net(x, h)
        else:
            out = net(x)
        z = torch.clamp(hist[:, -1] + out[:, :6], -Z_CLAMP, Z_CLAMP)
        if two_part:
            logits.append(out[:, 6])
            z = _gate_rain(z, out[:, 6], clim_targets[:, k], cfg["rain_ref"], mode)
        if relax is not None:
            z = z * relax
        outs.append(z)
        hist = torch.cat([hist[:, 1:], z[:, None]], dim=1)
    z_all = torch.stack(outs, dim=1)
    if return_logits:
        return z_all, (torch.stack(logits, dim=1) if logits else None)
    return z_all
