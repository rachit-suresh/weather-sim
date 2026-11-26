"""Network input construction, shared by training and inference (torch).

Per step the network sees (all on the 32 x 36 grid):
  18  the last 3 days as normalised anomalies           [3 x 6]
   6  climatology of the TARGET day, normalised          (so it knows the season's mean state)
   2  static maps: land/sea mask, orography (scaled)
   2  sin/cos day of year of the target day (broadcast)
   1  distance to the domain edge (scaled 0..1), for the limited-area boundary problem
  --
  29 channels (N_IN). Option F6 (spec EDG-2, `latlon`): + 2 maps (latitude, longitude scaled to -1..1) = 31.
The network predicts the change of the anomaly for the next day [6]; with the two-part rain head (RAIN-3)
it also outputs a 7th map, the logit of "rain > 1 mm/day".
"""
from __future__ import annotations

import numpy as np
import torch

from ..config import NLAT, NLON, T_IN

N_IN = T_IN * 6 + 6 + 2 + 2 + 1
N_OUT = 6
EDGE_RELAX_DEFAULT = (0.7, 0.85, 0.95)      # spec EDG-2 (F4): factors at distance 0, 1, 2 from the edge


def edge_cells() -> np.ndarray:
    """[H, W] integer distance (cells) to the nearest grid edge."""
    ii, jj = np.indices((NLAT, NLON))
    return np.minimum.reduce([ii, jj, NLAT - 1 - ii, NLON - 1 - jj])


def edge_distance() -> np.ndarray:
    return np.clip(edge_cells().astype("f4") / 8.0, 0.0, 1.0)[None]  # [1, H, W]


def edge_relax_map(factors) -> np.ndarray:
    """[H, W] multiplier applied to the anomaly after every step (1 away from the edge)."""
    d = edge_cells(); m = np.ones((NLAT, NLON), "f4")
    for k, f in enumerate(factors):
        m[d == k] = f
    return m


def latlon_maps() -> np.ndarray:
    ii, jj = np.indices((NLAT, NLON)).astype("f4")
    return np.stack([ii / (NLAT - 1) * 2 - 1, jj / (NLON - 1) * 2 - 1])     # [2, H, W]


def static_features(static: np.ndarray) -> np.ndarray:
    lsm = static[0]
    oro = static[1] / 1000.0                                           # km
    return np.stack([lsm, oro]).astype("f4")                           # [2, H, W]


def fixed_maps(static: np.ndarray, latlon: bool = False) -> np.ndarray:
    """Maps that never change: [lsm, oro_km, edge] (+ [lat, lon] if latlon) -> [3 or 5, H, W]."""
    parts = [static_features(static), edge_distance()] + ([latlon_maps()] if latlon else [])
    return np.concatenate(parts).astype("f4")


def n_inputs(latlon: bool = False) -> int:
    return N_IN + (2 if latlon else 0)


def make_input(hist: torch.Tensor, clim_target: torch.Tensor, sincos_target: torch.Tensor,
               fixed: torch.Tensor) -> torch.Tensor:
    """hist [B,3,6,H,W] anomalies, clim_target [B,6,H,W] normalised, sincos_target [B,2],
    fixed [3 or 5,H,W] (static 2 + edge 1 [+ lat, lon]) -> [B, n_inputs, H, W]."""
    b = hist.shape[0]
    sc = sincos_target[:, :, None, None].expand(b, 2, NLAT, NLON)
    fx = fixed[None].expand(b, -1, -1, -1)
    return torch.cat([hist.reshape(b, -1, NLAT, NLON), clim_target, fx[:, :2], sc, fx[:, 2:]], dim=1)
