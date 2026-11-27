"""Day and window flags for the round-F oversampling runs (spec V2-F-S, V2-F1, V2-F2). Training-only code.

    python -m training.oversample            # prints thresholds and flagged shares (calibration on 2005-2017)

cyclone     day flag: min msl anomaly (vs a 2005-2017 calendar-day climatology) over Bay of Bengal / Arabian Sea
            sea cells south of 23N is below CYCLONE_HPA
heavy_rain  day flag: number of cells with rain > 10 mm/day is >= its 90th percentile over 2005-2017 days
A window (last input day t, unroll k) is flagged if any target day t+1..t+k is flagged.
"""
from __future__ import annotations

import datetime as dt

import numpy as np

from twin.config import LAT, LON, MSL, TP

from .data import TRAIN_YEARS, Dataset, load, window_starts

CYCLONE_HPA = -8.0      # V2-F-S: about 3% of 2005-2017 days
HEAVY_MM = 10.0
HEAVY_PCT = 90.0        # top 10% of 2005-2017 days by heavy-rain area


def cyclone_region(static: np.ndarray) -> np.ndarray:
    sea = static[0] < 0.5
    la, lo = LAT[:, None], LON[None, :]
    bob = sea & (la <= 23) & (lo >= 80) & (lo <= 98)
    ars = sea & (la <= 23) & (lo <= 78)
    return bob | ars


def msl_min_anomaly_hpa(ds: Dataset) -> np.ndarray:
    """[N] minimum msl anomaly (hPa) over the cyclone region, climatology from 2005-2017 only."""
    from monsoon_r1.stats import _doy_index, climatology
    m = ds.mask(TRAIN_YEARS)
    clim = climatology(ds.data[m][:, [MSL]], ds.ymd[m], smooth_days=15)[:, 0]
    slot = np.array([_doy_index(dt.date(int(x) // 10000, int(x) // 100 % 100, int(x) % 100)) for x in ds.ymd])
    anom = (ds.data[:, MSL] - clim[slot]) / 100.0
    reg = cyclone_region(ds.static)
    return np.where(reg[None], anom, np.inf).reshape(len(anom), -1).min(axis=1)


def heavy_area(ds: Dataset) -> np.ndarray:
    return (ds.data[:, TP] > HEAVY_MM).sum(axis=(1, 2))


def day_flags(ds: Dataset, kind: str) -> tuple[np.ndarray, float]:
    """[N] bool day flags and the threshold used (calibrated on 2005-2017 days only)."""
    if kind == "cyclone":
        return msl_min_anomaly_hpa(ds) < CYCLONE_HPA, CYCLONE_HPA
    if kind == "heavy_rain":
        area = heavy_area(ds)
        q = float(np.percentile(area[ds.mask(TRAIN_YEARS)], HEAVY_PCT))
        return area >= q, q
    raise ValueError(kind)


def window_flags(day: np.ndarray, starts: np.ndarray, unroll: int) -> np.ndarray:
    """[len(starts)] True if any target day t+1..t+unroll of the window is flagged."""
    c = np.concatenate([[0], np.cumsum(day.astype(np.int64))])         # c[i] = flagged days before index i
    return (c[starts + unroll + 1] - c[starts + 1]) > 0


def main():
    ds = load(); m = ds.mask(TRAIN_YEARS)
    for kind in ("cyclone", "heavy_rain"):
        d, th = day_flags(ds, kind)
        shares = {u: round(float(window_flags(d, window_starts(ds, TRAIN_YEARS, u), u).mean()), 4) for u in (1, 2, 4, 8)}
        print(f"{kind}: threshold {th}  days flagged {d[m].mean():.4f} (2005-2017)  window share by unroll {shares}")


if __name__ == "__main__":
    main()
