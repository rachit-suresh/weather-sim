"""Shipped climatology ([366, 6, 32, 36], one year only) and static maps, loaded once."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from .calendar import clim_index
from .config import ARTIFACTS


@lru_cache(maxsize=None)
def load_climatology(path: Path | None = None) -> np.ndarray:
    z = np.load(path or ARTIFACTS / "climatology.npz")
    return z["clim"].astype("f4")


@lru_cache(maxsize=None)
def load_static(path: Path | None = None) -> np.ndarray:
    """[2, 32, 36]: land_sea_mask (0..1), orography (m)."""
    z = np.load(path or ARTIFACTS / "climatology.npz")
    return z["static"].astype("f4")


def clim_at(doy: float) -> np.ndarray:
    return load_climatology()[clim_index(doy)]


def clim_sequence(doy_last: float, steps: int) -> np.ndarray:
    """Climatology for forecast days 1..steps after the last input day."""
    c = load_climatology()
    return np.stack([c[clim_index(doy_last + k)] for k in range(1, steps + 1)])
