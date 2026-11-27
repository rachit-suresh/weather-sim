"""Heavy-rain frequency bias and raw CSI at > 10 mm/day (spec V2-F-S, F2 guard: every bias in 0.8-1.6).

    python -m training.heavy_bias --checkpoint models/<name>.pt --tag <name>

Uses the 60 random windows of val and test (training.events.random_windows). Bias = predicted / true number of cells
> 10 mm/day, pooled over windows, at lead days 1, 4 and 8. Appends one row per split to runs/heavy_bias.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from monsoon_r1 import contract as C

from twin.engine import Engine

from .data import ROOT, TEST_YEARS, VAL_YEARS, load
from .events import random_windows

HEAVY_MM = 10.0
LEADS = (1, 4, 8)
BAND = (0.8, 1.6)


def heavy_stats(eng: Engine, ds, years) -> dict:
    cnt = {L: np.zeros(3) for L in LEADS}                               # hits, predicted, true
    for t in random_windows(ds, years):
        pred = eng.simulate(ds.data[t - 2:t + 1], C.doy_sincos(int(ds.doy[t])), max(LEADS))
        for L in LEADS:
            p, o = pred[L - 1, 1] > HEAVY_MM, ds.data[t + L, 1] > HEAVY_MM
            cnt[L] += [(p & o).sum(), p.sum(), o.sum()]
    out = {}
    for L in LEADS:
        h, np_, nt = cnt[L]
        out[f"bias_d{L}"] = round(float(np_ / max(nt, 1)), 3)
        out[f"csi_d{L}"] = round(float(h / max(np_ + nt - h, 1)), 3)
    return out


def run(spec: dict, tag: str) -> list[dict]:
    ds = load(); eng = Engine(spec); rows = []
    for split, years in (("val", VAL_YEARS), ("test", TEST_YEARS)):
        r = {"time": datetime.now().isoformat(timespec="seconds"), "tag": tag, "model": eng.name, "split": split}
        r.update(heavy_stats(eng, ds, years))
        r["in_band"] = all(BAND[0] <= r[f"bias_d{L}"] <= BAND[1] for L in LEADS)
        rows.append(r)
    f = ROOT / "runs" / "heavy_bias.csv"; f.parent.mkdir(exist_ok=True)
    old = list(csv.DictReader(f.open(newline=""))) if f.exists() else []
    allr = old + rows; fields = list(dict.fromkeys(k for r in allr for k in r))
    with f.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval=""); w.writeheader(); w.writerows(allr)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(); g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--checkpoint"); g.add_argument("--spec")
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    spec = {"model": "neural", "checkpoint": a.checkpoint} if a.checkpoint else json.loads(Path(a.spec).read_text())
    for r in run(spec, a.tag or (a.checkpoint or a.spec)):
        print(json.dumps(r))


if __name__ == "__main__":
    main()
