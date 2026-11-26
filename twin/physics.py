"""Input checks and output clamps. Every rollout passes through `clean_output` before it leaves the server."""
from __future__ import annotations

import numpy as np

from .config import LIMITS, NC, NLAT, NLON


def clamp_state(x: np.ndarray) -> np.ndarray:
    """Clamp a [..., 6, 32, 36] physical state into safe ranges and replace NaN/Inf. Returns float32."""
    x = np.nan_to_num(np.asarray(x, dtype="f4"), nan=0.0, posinf=0.0, neginf=0.0)
    out = x.copy()
    for c, (lo, hi) in LIMITS.items():
        out[..., c, :, :] = np.clip(out[..., c, :, :], lo, hi)
    return out


def check_input(init_states: np.ndarray, perturbation: np.ndarray | None) -> None:
    if init_states.shape != (3, NC, NLAT, NLON):
        raise ValueError(f"init_states shape {init_states.shape} != (3, {NC}, {NLAT}, {NLON})")
    if perturbation is not None and perturbation.shape != (NC, NLAT, NLON):
        raise ValueError(f"perturbation shape {perturbation.shape} != ({NC}, {NLAT}, {NLON})")


def clean_output(rollout: np.ndarray, steps: int) -> np.ndarray:
    rollout = np.asarray(rollout, dtype="f4")
    if rollout.shape != (steps, NC, NLAT, NLON):
        raise RuntimeError(f"model returned shape {rollout.shape}, expected ({steps}, {NC}, {NLAT}, {NLON})")
    return clamp_state(rollout)
