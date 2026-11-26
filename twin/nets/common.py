"""Network input construction, shared by training and inference (torch).

Per step the network sees (all on the 32 x 36 grid):
  18  the last 3 days as normalised anomalies           [3 x 6]
   6  climatology of the TARGET day, normalised          (so it knows the season's mean state)
   2  static maps: land/sea mask, orography (scaled)
   2  sin/cos day of year of the target day (broadcast)
   1  distance to the domain edge (scaled 0..1), for the limited-area boundary problem
  --
  29 channels (N_IN). Option F6 (spec EDG-2, `latlon`): + 2 maps (latitude, longitude scaled to -1..1) = 31.
  Option `flux_conv` (spec V2-E3): + 1 map, moisture-flux convergence of the current state (built in rollout.py).
The network predicts the change of the anomaly for the next day [6]; with the two-part rain head (RAIN-3)
it also outputs a 7th map, the logit of "rain > 1 mm/day".
"""
from __future__ import annotations

import numpy as np
import torch

from ..config import LAT, NLAT, NLON, T_IN

N_IN = T_IN * 6 + 6 + 2 + 2 + 1
N_OUT = 6
EDGE_RELAX_DEFAULT = (0.7, 0.85, 0.95)      # spec EDG-2 (F4): factors at distance 0, 1, 2 from the edge
DEG_M = 111.2e3                             # metres per degree of latitude (grid spacing 1 degree)
FLUX_CLAMP = 6.0                            # spec V2-E3: normalised convergence clamped to +-6


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


def n_inputs(latlon: bool = False, flux_conv: bool = False) -> int:
    return N_IN + (2 if latlon else 0) + (1 if flux_conv else 0)


def flux_convergence(u: torch.Tensor, v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Spec V2-E3. u, v [.., H, W] in m/s, q (tcwv) in kg/m2 on the 1-degree grid (row 0 = southernmost latitude)
    -> C = -div(q V) in mm/day (kg/m2/day). Centred differences over 2 cells, replicate edges (edge cells use
    (f[1] - f[0]) / 2 cells); plus-shaped stencil: each output cell reads only its 4 direct neighbours."""
    fu, fv = q * u, q * v
    lead = fu.shape[:-2]
    fu = fu.reshape(-1, 1, NLAT, NLON); fv = fv.reshape(-1, 1, NLAT, NLON)
    pu = torch.nn.functional.pad(fu, (1, 1, 0, 0), mode="replicate")
    pv = torch.nn.functional.pad(fv, (0, 0, 1, 1), mode="replicate")
    coslat = torch.as_tensor(np.cos(np.deg2rad(LAT)), dtype=u.dtype, device=u.device)[:, None]
    dfu = (pu[..., :, 2:] - pu[..., :, :-2]) / (2 * DEG_M * coslat)
    dfv = (pv[..., 2:, :] - pv[..., :-2, :]) / (2 * DEG_M)
    return (-(dfu + dfv) * 86400.0).reshape(*lead, NLAT, NLON)


def make_input(hist: torch.Tensor, clim_target: torch.Tensor, sincos_target: torch.Tensor,
               fixed: torch.Tensor) -> torch.Tensor:
    """hist [B,3,6,H,W] anomalies, clim_target [B,6,H,W] normalised, sincos_target [B,2],
    fixed [3 or 5,H,W] (static 2 + edge 1 [+ lat, lon]) -> [B, n_inputs, H, W]."""
    b = hist.shape[0]
    sc = sincos_target[:, :, None, None].expand(b, 2, NLAT, NLON)
    fx = fixed[None].expand(b, -1, -1, -1)
    return torch.cat([hist.reshape(b, -1, NLAT, NLON), clim_target, fx[:, :2], sc, fx[:, 2:]], dim=1)
