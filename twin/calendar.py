"""Day-of-year helpers. The API only sends [sin, cos] of the last input day; years are never known."""
from __future__ import annotations

import math

import numpy as np

YEAR = 365.0


def doy_from_sincos(sincos) -> float:
    """[sin(2 pi d/365), cos(2 pi d/365)] -> d in [0, 365)."""
    s, c = float(sincos[0]), float(sincos[1])
    return (math.atan2(s, c) / (2 * math.pi) * YEAR) % YEAR


def sincos(doy: float) -> np.ndarray:
    a = 2 * math.pi * doy / YEAR
    return np.array([math.sin(a), math.cos(a)], "f4")


def clim_index(doy: float) -> int:
    """Index into a [366, ...] climatology built on a leap-year calendar (slot 59 = 29 Feb, as in
    monsoon_r1.stats). The API day count has no leap day, so days from 1 Mar onward skip slot 59.
    Off-by-one in the caller's day convention only shifts a 15-day-smoothed field by one day."""
    k = int(round(doy)) % 365
    return k + 1 if k >= 59 else k
