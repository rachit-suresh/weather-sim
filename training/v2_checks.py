"""v2 checks that look past the judge's capped scores (spec V2-P). Same yardstick for every v2 experiment.

    python -m training.v2_checks --checkpoint models/<name>.pt [--tag <name>]
    python -m training.v2_checks --spec path/to/spec.json --tag <name>

Reports (and appends to runs/v2_checks.csv):
  1. Cyclone track (validation year): Cyclone Fani, last input 1 May 2019 -> centre error (km) days 1-3 and the
     latitude error on landfall day (+2). Amphan (test year 2020) is reported too but must NOT drive decisions.
  2. Wind -> rain response: the judge's monsoon_stall shock (-40% u10/v10 over 8-20N, 65-75E) on 12 monsoon windows
     from validation years; mean rain change (days 1-3) over west-coast land (8-20N, 72-77E). Physics expects LESS rain
     (weaker onshore flow), so the sign should be negative; reports mean change and share of windows < 0.
  3. Edge-band error: day-4 RMSE (all channels, std units) in the outer 3-cell ring vs the interior, validation windows.
  4. Speed: ms per 10-day rollout.
Uncapped skill / CSI / shocks / reach come from training.evaluate (leaderboard.csv): use those columns, not points.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from monsoon_r1 import contract as C
from monsoon_r1 import scoring as S

from twin.climatology import load_static
from twin.config import LAT, LON, load_stats
from twin.engine import Engine
from twin.nets.common import edge_cells

from .data import ROOT, VAL_YEARS, load, window_starts

SEARCH = ((LAT >= 8) & (LAT <= 27))[:, None] & ((LON >= 80) & (LON <= 95))[None, :]
WEST_COAST = ((LAT >= 8) & (LAT <= 20))[:, None] & ((LON >= 72) & (LON <= 77))[None, :]


def _centre(state):
    m = np.where(SEARCH, state[4], np.inf); i, j = np.unravel_index(np.argmin(m), m.shape)
    return float(LAT[i]), float(LON[j]), float(state[4][i, j] / 100)


def _km(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    return 6371 * 2 * math.asin(math.sqrt(math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2))


def track(eng, ds, last_ymd):
    t = int(np.where(ds.ymd == last_ymd)[0][0]); init = ds.data[t - 2:t + 1]
    pred = eng.simulate(init, C.doy_sincos(int(ds.doy[t])), 3)
    real = [_centre(ds.data[t + k]) for k in (1, 2, 3)]; mod = [_centre(pred[k]) for k in range(3)]
    return {"err_km": [round(_km(r, m)) for r, m in zip(real, mod)], "landfall_lat_err": round(mod[1][0] - real[1][0], 1),
            "real": real, "model": mod}


def wind_rain(eng, ds, std, n=12):
    lsm = load_static()[0] > 0.5; reg = WEST_COAST & lsm
    idx = window_starts(ds, VAL_YEARS, 3)
    idx = [t for t in idx if 6 <= ds.months[t] <= 9]
    rng = np.random.default_rng(0); idx = sorted(rng.choice(idx, size=min(n, len(idx)), replace=False))
    deltas = []
    for t in idx:
        init = ds.data[t - 2:t + 1]; sc = C.doy_sincos(int(ds.doy[t]))
        base = eng.simulate(init, sc, 3); shock = eng.simulate(init, sc, 3, S.make_shock("monsoon_stall", init[-1], std))
        deltas.append(float((shock[:, 1] - base[:, 1])[:, reg].mean()))
    d = np.array(deltas)
    return {"mean_rain_change_mm": round(float(d.mean()), 3), "share_negative": round(float((d < 0).mean()), 2)}


def edge_band(eng, ds, std, n=30):
    ring = edge_cells() < 3
    idx = window_starts(ds, VAL_YEARS, 4); rng = np.random.default_rng(1); idx = rng.choice(idx, size=n, replace=False)
    e_ring, e_in, ms = [], [], []
    for t in idx:
        init = ds.data[t - 2:t + 1]; t0 = time.perf_counter()
        p = eng.simulate(init, C.doy_sincos(int(ds.doy[t])), 10); ms.append((time.perf_counter() - t0) * 1000)
        err = ((p[3] - ds.data[t + 4]) / std[:, None, None]) ** 2
        e_ring.append(err[:, ring].mean()); e_in.append(err[:, ~ring].mean())
    return {"edge_rmse_std": round(float(np.sqrt(np.mean(e_ring))), 4), "interior_rmse_std": round(float(np.sqrt(np.mean(e_in))), 4),
            "ms_per_rollout": round(float(np.mean(ms)), 1)}


def run(spec: dict, tag: str) -> dict:
    ds = load(); eng = Engine(spec); _, std = load_stats()
    res = {"time": datetime.now().isoformat(timespec="seconds"), "tag": tag, "model": eng.name}
    fani = track(eng, ds, 20190501); amphan = track(eng, ds, 20200518)
    res.update({"fani_err_km_d1": fani["err_km"][0], "fani_err_km_d2": fani["err_km"][1], "fani_err_km_d3": fani["err_km"][2],
                "fani_landfall_lat_err": fani["landfall_lat_err"],
                "amphan_err_km_d2_TESTONLY": amphan["err_km"][1], "amphan_landfall_lat_err_TESTONLY": amphan["landfall_lat_err"]})
    res.update(wind_rain(eng, ds, std)); res.update(edge_band(eng, ds, std))
    f = ROOT / "runs" / "v2_checks.csv"; f.parent.mkdir(exist_ok=True)
    rows = list(csv.DictReader(f.open(newline=""))) if f.exists() else []
    rows.append(res); fields = list(dict.fromkeys(k for r in rows for k in r))
    with f.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval=""); w.writeheader(); w.writerows(rows)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(); g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--checkpoint"); g.add_argument("--spec")
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    spec = {"model": "neural", "checkpoint": a.checkpoint} if a.checkpoint else json.loads(Path(a.spec).read_text())
    print(json.dumps(run(spec, a.tag or (a.checkpoint or a.spec)), indent=1))


if __name__ == "__main__":
    main()
