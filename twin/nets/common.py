"""Network input construction, shared by training and inference (torch).

Per step the network sees (all on the 32 x 36 grid):
  18  the last 3 days as normalised anomalies           [3 x 6]
   6  climatology of the TARGET day, normalised          (so it knows the season's mean state)
   2  static maps: land/sea mask, orography (scaled)
   2  sin/cos day of year of the target day (broadcast)
   1  distance to the domain edge (scaled 0..1), for the limited-area boundary problem
  --
  29 channels in total -> the network predicts the change of the anomaly for the next day [6].
"""
from __future__ import annotations

import numpy as np
import torch

from ..config import NLAT, NLON, T_IN

N_IN = T_IN * 6 + 6 + 2 + 2 + 1
N_OUT = 6


def edge_distance() -> np.ndarray:
    ii, jj = np.indices((NLAT, NLON))
    d = np.minimum.reduce([ii, jj, NLAT - 1 - ii, NLON - 1 - jj]).astype("f4")
    return np.clip(d / 8.0, 0.0, 1.0)[None]                           # [1, H, W]


def static_features(static: np.ndarray) -> np.ndarray:
    lsm = static[0]
    oro = static[1] / 1000.0                                           # km
    return np.stack([lsm, oro]).astype("f4")                           # [2, H, W]


def make_input(hist: torch.Tensor, clim_target: torch.Tensor, sincos_target: torch.Tensor,
               fixed: torch.Tensor) -> torch.Tensor:
    """hist [B,3,6,H,W] anomalies, clim_target [B,6,H,W] normalised, sincos_target [B,2],
    fixed [3,H,W] (static 2 + edge 1) -> [B, N_IN, H, W]."""
    b = hist.shape[0]
    sc = sincos_target[:, :, None, None].expand(b, 2, NLAT, NLON)
    fx = fixed[None].expand(b, -1, -1, -1)
    return torch.cat([hist.reshape(b, -1, NLAT, NLON), clim_target, fx[:, :2], sc, fx[:, 2:]], dim=1)
