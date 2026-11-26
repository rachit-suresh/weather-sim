"""Cyclone-mode calibration (spec V2-CYC b, d). Training-only code.

    python -m training.cyclone_calib

1. Threshold: 98th percentile over TRAINING days (2005-2017) of the basin minimum -(msl anomaly). Val/test years
   are only used to report how often the detector fires, never to set anything.
2. How often it fires on train / val / test days, and on the evaluate windows (events / random / named).
3. Composite radial profile of the detected training lows (msl anomaly / centre value, wind-speed anomaly), which
   is the reasoning behind the mask radius R = 4 + taper 2 cells.
4. LOST (track-lost level, V2-CYC c) = training median of the same basin minimum.
5. --tracker: tracker check on the TRAINING episodes (first detected day of each episode): mean mask weight, days
   1-3, at the mover's low (deepest mover-minus-base msl difference within 9 cells of the start) and at the true low.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from twin.calendar import clim_index
from twin.climatology import load_climatology
from twin.config import LAT, LON, MSL, U10, V10
from twin.models import cyclone as CY
from twin.models.cyclone import LOST_PA, THRESHOLD_PA, basin_cells

from .data import TEST_YEARS, TRAIN_YEARS, VAL_YEARS, load
from .events import named_windows, pick_events, random_windows

PCT = 98.0


def basin_min(ds) -> tuple[np.ndarray, np.ndarray]:
    """Per day: basin minimum of the msl anomaly (Pa) and the flat argmin cell."""
    clim = load_climatology()[[clim_index(d) for d in ds.doy]]
    a = np.where(basin_cells(), ds.data[:, MSL] - clim[:, MSL], np.inf).reshape(len(ds.data), -1)
    return a.min(1), a.argmin(1)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--tracker", action="store_true"); args = ap.parse_args(argv)
    ds = load()
    mn, arg = basin_min(ds)
    tr = ds.mask(TRAIN_YEARS)
    thr = float(np.percentile(-mn[tr], PCT))
    print(f"threshold (P{PCT:.0f} of training-day basin minimum): {thr:.1f} Pa = {thr / 100:.2f} hPa;"
          f" serving constant THRESHOLD_PA = {THRESHOLD_PA}")
    lost = float(np.percentile(-mn[tr], 50))
    print(f"LOST (P50 of the same): {lost:.1f} Pa; serving constant LOST_PA = {LOST_PA}")
    fire = -mn > THRESHOLD_PA
    rates = {}
    for lab, yrs in (("train", TRAIN_YEARS), ("val", VAL_YEARS), ("test", TEST_YEARS)):
        m = ds.mask(yrs)
        rates[lab] = {"days": int(m.sum()), "fires": int((fire & m).sum()), "share": round(float((fire & m).mean() / m.mean()), 4)}
    print("fires on days:", json.dumps(rates))
    for lab, yrs, split in (("val", VAL_YEARS, "val"), ("test", TEST_YEARS, "test")):
        ev = pick_events(ds, yrs); nm = named_windows(ds, split); rnd = random_windows(ds, yrs)
        hit = lambda t: bool(fire[t])
        print(f"{lab}: events fired {[k for k, t in ev.items() if hit(t)]} of {list(ev)}")
        print(f"{lab}: named fired {[k for k, t in nm.items() if hit(t)]} of {list(nm)}")
        print(f"{lab}: random fired {sum(hit(t) for t in rnd)} of {len(rnd)}")
        for k, t in {**ev, **nm}.items():
            i, j = np.unravel_index(arg[t], (len(LAT), len(LON)))
            print(f"    {k:20s} last input {ds.ymd[t]}  basin min {mn[t] / 100:6.1f} hPa at {LAT[i]}N {LON[j]}E")
    # composite profile of the detected training lows
    clim = load_climatology()
    ii, jj = np.indices((len(LAT), len(LON)))
    prof, wind = {r: [] for r in range(9)}, {r: [] for r in range(9)}
    for t in np.where(tr & fire)[0]:
        c = clim[clim_index(ds.doy[t])]
        a = ds.data[t, MSL] - c[MSL]
        sp = np.hypot(ds.data[t, U10], ds.data[t, V10]) - np.hypot(c[U10], c[V10])
        i, j = np.unravel_index(arg[t], a.shape)
        d = np.round(np.hypot(ii - i, jj - j)).astype(int)
        for r in prof:
            prof[r].append(a[d == r].mean() / a[i, j]); wind[r].append(sp[d == r].mean())
    print("composite of detected training lows: radius (cells), msl anomaly / centre, wind-speed anomaly (m/s)")
    for r in prof:
        print(f"    {r}  {np.mean(prof[r]):.3f}  {np.mean(wind[r]):+.2f}")
    if args.tracker:
        tracker_check(ds, tr & fire)


def tracker_check(ds, fire):
    from twin.models.neural import NeuralModel
    mover, base = NeuralModel("models/v2_e7c.pt"), NeuralModel("models/r3_cnn.pt")
    ii, jj = np.indices((len(LAT), len(LON)))
    starts = [t for t in np.where(fire)[0] if not fire[t - 1]]
    runs = {}
    for t in starts:
        init, doy = ds.data[t - 2:t + 1], float(ds.doy[t])
        runs[t] = (CY.detect(init[-1], doy), mover.rollout(init, doy, 3), base.rollout(init, doy, 3))
    print(f"tracker check on {len(starts)} training episodes: search (cells), lost (Pa) -> mean mask weight at the"
          " mover's low / at the true low, mask cells")
    for search, lost in ((4, THRESHOLD_PA / 2), (4, LOST_PA), (6, LOST_PA), (7, THRESHOLD_PA / 2), (7, LOST_PA)):
        w_mov, w_true, area = [], [], []
        for t, (st, free, r3) in runs.items():
            doy = float(ds.doy[t])
            m = CY.swath_masks(CY.track(st, free, doy, lost, search))
            truth = CY.track(st, ds.data[t + 1:t + 4], doy, 0.0, 7.0)
            near = np.hypot(ii - st[0], jj - st[1]) <= 9
            for k in range(3):
                d = np.where(near, free[k, MSL] - r3[k, MSL], np.inf)
                w_mov.append(m[k][np.unravel_index(np.argmin(d), d.shape)]); w_true.append(m[k][truth[k + 1]])
                area.append((m[k] > 0).sum())
        print(f"    {search}, {lost:6.1f} -> {np.mean(w_mov):.2f} / {np.mean(w_true):.2f}, {np.mean(area):.0f}")


if __name__ == "__main__":
    main()
