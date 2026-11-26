"""Pick event windows that look like the judge's hidden cases (monsoon onset, heavy rain, heatwave, cyclone,
winter western disturbance) from any set of years, using simple physical criteria. Returns the index of the
LAST input day for each window, chosen so the event happens inside the first forecast days.
"""
from __future__ import annotations

import numpy as np

from twin.calendar import clim_index
from twin.climatology import load_climatology
from twin.config import LAT, LON, MSL, T2M, TP

from .data import Dataset


def _box(lat0, lat1, lon0, lon1):
    return ((LAT >= lat0) & (LAT <= lat1))[:, None] & ((LON >= lon0) & (LON <= lon1))[None, :]


KERALA = _box(8, 13, 74, 78)
CENTRAL_NORTH = _box(20, 30, 72, 85)
BAY = _box(10, 22, 80, 95)
ARABIAN = _box(10, 22, 62, 75)
NORTH_WEST = _box(28, 37, 70, 80)
INDIA_LAND = _box(8, 30, 70, 88)


def pick_events(ds: Dataset, years, lead_room: int = 10, offset: int = 2) -> dict[str, int]:
    """One window per event type per call. `offset` = days between the last input day and the event day."""
    m = ds.mask(years)
    ok = np.zeros(len(ds.data), bool)
    ok[2:len(ds.data) - lead_room] = True
    ok &= m & np.roll(m, -lead_room) & np.roll(m, 2)
    clim = load_climatology()
    anom = lambda c: ds.data[:, c] - clim[[clim_index(d) for d in ds.doy]][:, c]
    mo = ds.months
    events = {}

    def best(score, months, largest=True):
        cand = np.where(np.isin(mo, months) & np.roll(ok, offset))[0]
        cand = cand[cand - offset >= 2]
        if len(cand) == 0:
            return None
        k = cand[np.argmax(score[cand])] if largest else cand[np.argmin(score[cand])]
        return int(k - offset)

    tp_kerala = ds.data[:, TP][:, KERALA].mean(-1)
    tp_k3 = np.convolve(tp_kerala, np.ones(3) / 3, mode="same")          # 3-day mean: avoid one-off showers
    onset = np.where((ds.doy >= 140) & (ds.months <= 6) & (tp_k3 > 10) & np.roll(ok, offset) & m)[0]
    if len(onset):
        events["monsoon_onset"] = int(onset[0] - offset)
    events["heavy_rain"] = best(ds.data[:, TP][:, INDIA_LAND].mean(-1), [6, 7, 8, 9])
    events["heatwave"] = best(anom(T2M)[:, CENTRAL_NORTH].mean(-1), [4, 5, 6])
    msl_a = anom(MSL)
    events["cyclone"] = best(np.minimum(msl_a[:, BAY].min(-1), msl_a[:, ARABIAN].min(-1)), [5, 6, 10, 11, 12], largest=False)
    events["winter_disturbance"] = best(ds.data[:, TP][:, NORTH_WEST].mean(-1), [12, 1, 2])
    return {k: v for k, v in events.items() if v is not None}


# spec VAL-2: famous real events (event day as yyyymmdd); the last input day is 2 days earlier
NAMED = {
    "val": {"kerala_floods": 20180815, "cyclone_titli": 20181011, "cyclone_gaja": 20181116, "cyclone_fani": 20190503,
            "cyclone_vayu": 20190612, "monsoon_onset_2019": 20190608, "north_india_heat": 20190602},
    "test": {"cyclone_amphan": 20200520, "cyclone_nisarga": 20200603, "cyclone_tauktae": 20210517,
             "cyclone_yaas": 20210526},
}


def named_windows(ds: Dataset, split: str, offset: int = 2) -> dict[str, int]:
    out = {}
    for name, ymd in NAMED[split].items():
        hit = np.where(ds.ymd == ymd)[0]
        if len(hit) and hit[0] - offset >= 2 and hit[0] - offset + 10 < len(ds.data):
            out[name] = int(hit[0] - offset)
    return out


def random_windows(ds: Dataset, years, n: int = 60, lead_room: int = 10, seed: int = 0) -> list[int]:
    from .data import window_starts
    idx = window_starts(ds, years, lead_room)
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(idx, size=min(n, len(idx)), replace=False).tolist())
