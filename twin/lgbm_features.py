"""Per-cell tabular features for the LightGBM family (shared by training and inference, numpy only).

For each cell and each forecast day, from normalised anomalies:
  54  last day, 3x3 neighbourhood x 6 channels   (locality: 1 cell per step)
  12  the 2 earlier days at the cell itself        (trend)
   6  climatology feature of the target day at the cell
   5  land/sea, orography, sin/cos day of year, distance to edge
  --
  77 features -> one model per channel predicts the next-day change of the anomaly.
"""
from __future__ import annotations

import numpy as np

N_FEAT = 54 + 12 + 6 + 5


def _patch(z: np.ndarray) -> np.ndarray:
    """[D,6,H,W] -> [D,54,H,W]: the 3x3 neighbourhood of every cell (edges replicated)."""
    h, w = z.shape[-2:]
    p = np.pad(z, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="edge")
    return np.concatenate([p[:, :, 1 + dy:1 + dy + h, 1 + dx:1 + dx + w] for dy in (-1, 0, 1) for dx in (-1, 0, 1)], axis=1)


def features(hist: np.ndarray, clim_target: np.ndarray, sincos_target: np.ndarray, fixed: np.ndarray) -> np.ndarray:
    """hist [D,3,6,H,W]; clim_target [D,6,H,W]; sincos_target [D,2]; fixed [3,H,W] (lsm, oro_km, edge)
    -> rows [D*H*W, N_FEAT] float32 (row order: day, lat, lon)."""
    d, _, _, h, w = hist.shape
    sc = np.broadcast_to(sincos_target[:, :, None, None], (d, 2, h, w))
    fx = np.broadcast_to(fixed[None], (d, 3, h, w))
    f = np.concatenate([_patch(hist[:, 2]), hist[:, 0], hist[:, 1], clim_target, fx[:, :2], sc, fx[:, 2:]], axis=1)
    return np.ascontiguousarray(f.transpose(0, 2, 3, 1).reshape(-1, N_FEAT), dtype="f4")
