"""Cyclone mode (spec V2-CYC): a stationary-low base model everywhere, a low-moving model inside a moving circle.

    {"model": "cyclone_mode", "base": "models/r3_cnn.pt", "mover": "models/v2_e7c.pt", "variant": "A"}

1. Detector: the UNPERTURBED last input day has a deep msl low (anomaly vs climatology < -THR) over the sea in
   the Bay of Bengal or the Arabian Sea.
2. Track: the mover is run with zero perturbation; its low is followed day by day (V2-CYC c).
3. Mask: the swath of cosine-tapered circles around the track so far (V2-CYC d). It depends only on the
   unperturbed states, so the shocked and the unshocked request share it (null test exact, reach <= max of both).
4. Output: mask * mover + (1 - mask) * base on the variant's channels; base elsewhere. No detection -> base,
   bit for bit.
"""
from __future__ import annotations

import numpy as np

from ..climatology import clim_at, load_static
from ..config import LAT, LON, MSL, NC, U10, V10
from .base import TwinModel
from .neural import NeuralModel

# V2-CYC b: 98th percentile of the training-day basin minimum of -(msl anomaly), from training/cyclone_calib.py
THRESHOLD_PA = 917.6
LOST_PA = 192.1         # V2-CYC c: training median of the same basin minimum (an ordinary day's deepest low)
RADIUS = 4.0            # full weight up to R cells (V2-CYC d)
TAPER = 2.0             # cosine taper to zero over T more cells
SEARCH = 7.0            # tracker search radius, cells per day (upper bound on a cyclone's daily translation)
VARIANTS = {"A": (U10, V10, MSL), "B": tuple(range(NC))}

_II, _JJ = np.indices((len(LAT), len(LON)))


def basin_cells() -> np.ndarray:
    """[32, 36] bool: sea cells (land_sea_mask < 0.5), lat <= 24N, lon 80-97E (Bay) or lon <= 77E (Arabian Sea)."""
    sea = load_static()[0] < 0.5
    lat, lon = LAT[:, None], LON[None, :]
    return sea & (lat <= 24) & (((lon >= 80) & (lon <= 97)) | (lon <= 77))


def msl_anomaly(state: np.ndarray, doy: float) -> np.ndarray:
    return state[MSL] - clim_at(doy)[MSL]


def detect(last_state: np.ndarray, doy: float, threshold: float = THRESHOLD_PA):
    """(i, j) of the basin low if it is deeper than -threshold, else None."""
    a = np.where(basin_cells(), msl_anomaly(last_state, doy), np.inf)
    i, j = np.unravel_index(np.argmin(a), a.shape)
    return (int(i), int(j)) if a[i, j] < -threshold else None


def track(start, pred: np.ndarray, doy: float, lost_pa: float = LOST_PA, search: float = SEARCH) -> list:
    """Centres c_0 (start) .. c_steps of the low in `pred` ([steps, 6, 32, 36], days doy+1 ..). Once the low is
    weaker than -lost_pa the track is lost and the centre stays where it was."""
    cs, lost = [start], False
    for k in range(len(pred)):
        ci, cj = cs[-1]
        if not lost:
            a = msl_anomaly(pred[k], doy + k + 1)
            a = np.where(np.hypot(_II - ci, _JJ - cj) <= search, a, np.inf)
            i, j = np.unravel_index(np.argmin(a), a.shape)
            lost = a[i, j] > -lost_pa
            if not lost:
                ci, cj = int(i), int(j)
        cs.append((ci, cj))
    return cs


def circle(c) -> np.ndarray:
    d = np.hypot(_II - c[0], _JJ - c[1])
    w = 0.5 * (1 + np.cos(np.pi * np.clip(d - RADIUS, 0, TAPER) / TAPER))
    return np.where(d >= RADIUS + TAPER, 0.0, w)


def swath_masks(centres: list) -> np.ndarray:
    """[steps, 32, 36]: day-k mask = max over j <= k of circle(c_j) (centres[0] is the start)."""
    m = circle(centres[0])
    out = []
    for c in centres[1:]:
        m = np.maximum(m, circle(c))
        out.append(m)
    return np.stack(out).astype("f4")


class CycloneMode(TwinModel):
    def __init__(self, base: str, mover: str, variant: str = "A", threshold: float = THRESHOLD_PA):
        self.base, self.mover = NeuralModel(base), NeuralModel(mover)
        self.channels = list(VARIANTS[variant])
        self.threshold = float(threshold)
        self.name = f"cyclone_mode_{variant}:{self.base.name.split(':')[-1]}+{self.mover.name.split(':')[-1]}"

    def rollout(self, init_states, doy, steps):
        return self.rollout_shocked(init_states, init_states, doy, steps)

    def masks(self, base_states, doy, steps, mover_free=None):
        """(masks [steps, 32, 36], mover's zero-perturbation forecast) or (None, None) when the detector is silent."""
        start = detect(base_states[-1], doy, self.threshold)
        if start is None:
            return None, None
        free = self.mover.rollout(base_states, doy, steps) if mover_free is None else mover_free
        return swath_masks(track(start, free, doy)), free

    def rollout_shocked(self, init_states, base_states, doy, steps):
        r3 = self.base.rollout(init_states, doy, steps)
        m, free = self.masks(base_states, doy, steps)
        if m is None:
            return r3
        e7 = free if np.array_equal(init_states, base_states) else self.mover.rollout(init_states, doy, steps)
        out = r3.copy()
        for c in self.channels:
            out[:, c] = m * e7[:, c] + (1 - m) * r3[:, c]
        return out

    def warmup(self):
        self.base.warmup(); self.mover.warmup()
