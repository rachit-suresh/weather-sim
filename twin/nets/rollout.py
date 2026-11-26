"""Autoregressive rollout in normalised-anomaly space, used by BOTH training (with gradients) and inference.

Stateless nets: out = net(x).  Stateful nets (`stateful = True`, e.g. ConvGRU): out, h = net(x, h); the hidden
state starts as None on every call, so each request is still a pure function of its inputs (null-shock test).

Options (dict `cfg`, stored in the checkpoint as "rollout"):
  edge_relax  [f0, f1, f2]  spec EDG-2 (F4): multiply the anomaly of cells 0/1/2 cells from the edge after every step
  rain_head   "two_part"    spec RAIN-3 (D4): net output channel 6 is the logit of "rain > 1 mm/day"
  rain_ref    {mean, std, anom_std}  global mean/std of rain (mm/day) and the rain anomaly std (sqrt space),
                            needed to turn the gate into anomaly units (rain must use the sqrt transform)
  flux_conv   {mean, std, anom_std, scale}  spec V2-E3: u10/v10/tcwv global mean/std, their anomaly std, and the
                            training-set std of the convergence (mm/day). Every step appends one input map:
                            -div(tcwv V) of the current state / scale, clamped to +-6. The current state is decoded with
                            the climatology of its own day: `clim_init` (last input day) for step 1, then the previous
                            target day; without `clim_init` step 1 uses the first target day's.
Rain gate (spec 1.2): hard (amount where logit > 0, else 0) in both modes; in mode "train" gradients flow as if the
gate were sigmoid(logit) (straight-through), so training and inference see the same rain.
"""
from __future__ import annotations

import torch

from ..config import TCWV, TP, U10, V10
from .common import FLUX_CLAMP, edge_relax_map, flux_convergence, make_input

Z_CLAMP = 12.0     # anomalies beyond 12 std are unphysical: clamp to keep long rollouts stable


def _gate_rain(z, logit, clim_k, ref, mode):
    """Replace the rain anomaly by the gated amount. clim_k [B,6,H,W] = target-day climatology feature."""
    c = torch.sqrt(torch.relu(clim_k[:, TP] * ref["std"] + ref["mean"]))          # sqrt(climatological rain)
    amount = torch.relu(z[:, TP] * ref["anom_std"] + c)                           # sqrt(rain) >= 0
    hard = (logit > 0).to(z.dtype)
    if mode == "train":                                       # straight-through: forward = hard, gradient = sigmoid
        soft = torch.sigmoid(logit)
        g = hard + soft - soft.detach()
    else:
        g = hard
    z_tp = (g * amount - c) / ref["anom_std"]
    return torch.cat([z[:, :TP], z_tp[:, None], z[:, TP + 1:]], dim=1)


def decode_wind_water(z: torch.Tensor, clim: torch.Tensor, ref: dict):
    """z [...,6,H,W] anomalies, clim [...,6,H,W] normalised climatology of the SAME day -> u10, v10 (m/s), tcwv (kg/m2)."""
    out = []
    for i, ch in enumerate((U10, V10, TCWV)):
        out.append(z[..., ch, :, :] * ref["anom_std"][i] + clim[..., ch, :, :] * ref["std"][i] + ref["mean"][i])
    return out[0], out[1], torch.relu(out[2])


def flux_map(z: torch.Tensor, clim: torch.Tensor, ref: dict) -> torch.Tensor:
    """Spec V2-E3 input map: [B,6,H,W] current state anomalies + its day's climatology -> [B,1,H,W]."""
    c = flux_convergence(*decode_wind_water(z, clim, ref)) / ref["scale"]
    return torch.clamp(c, -FLUX_CLAMP, FLUX_CLAMP)[:, None]


def rollout(net, hist: torch.Tensor, clim_targets: torch.Tensor, sincos_targets: torch.Tensor,
            fixed: torch.Tensor, steps: int, cfg: dict | None = None, mode: str = "train", return_logits: bool = False,
            clim_init: torch.Tensor | None = None):
    """hist [B,3,6,H,W]; clim_targets [B,steps,6,H,W]; sincos_targets [B,steps,2] -> [B,steps,6,H,W] anomalies
    (and [B,steps,H,W] rain logits if return_logits and the two-part head is on). clim_init [B,6,H,W] = normalised
    climatology of the last input day (only used by the flux_conv option)."""
    cfg = cfg or {}
    stateful = getattr(net, "stateful", False)
    two_part = cfg.get("rain_head") == "two_part"
    relax = None
    if cfg.get("edge_relax"):
        relax = torch.as_tensor(edge_relax_map(cfg["edge_relax"]), dtype=hist.dtype, device=hist.device)
    flux = cfg.get("flux_conv")
    h = None
    outs, logits = [], []
    for k in range(steps):
        x = make_input(hist, clim_targets[:, k], sincos_targets[:, k], fixed)
        if flux:
            clim_now = clim_targets[:, k - 1] if k > 0 else (clim_init if clim_init is not None else clim_targets[:, 0])
            x = torch.cat([x, flux_map(hist[:, -1], clim_now, flux)], dim=1)
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
