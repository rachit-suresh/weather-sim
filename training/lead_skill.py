"""Per-lead skill curve (spec V2-F-M): SS at lead L vs min(persistence, climatology), channel mean, on the 60
validation random windows; plus paired differences against a reference model with 2-standard-error noise bars.

    python -m training.lead_skill --checkpoints models/r3_cnn.pt models/v2_f4.pt models/v2_f5.pt [--ref r3_cnn]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from monsoon_r1 import contract as C
from monsoon_r1 import scoring as S

from twin.climatology import clim_sequence
from twin.engine import Engine

from .data import TEST_YEARS, VAL_YEARS, load
from .events import random_windows

LEADS = [1, 2, 4, 6, 8, 10]


def per_window(spec: dict, split: str = "val") -> np.ndarray:
    """[n_windows, len(LEADS)] channel-mean ss at each lead (judge's skill_vs_best_ref at that lead alone)."""
    ds = load(); eng = Engine(spec)
    idx = random_windows(ds, VAL_YEARS if split == "val" else TEST_YEARS)
    out = []
    for t in idx:
        init, doy, truth = ds.data[t - 2:t + 1], ds.doy[t], ds.data[t + 1:t + 11]
        pred = eng.simulate(init, C.doy_sincos(int(doy)), 10)
        persist = np.repeat(init[-1][None], 10, axis=0); clim = clim_sequence(float(doy), 10)
        out.append([float(np.nanmean(S.skill_vs_best_ref(pred[[L - 1]], truth[[L - 1]], persist[[L - 1]],
                                                         clim[[L - 1]])["ss"])) for L in LEADS])
    return np.array(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--ref", default=None, help="stem of the reference model for paired differences")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--out", default=None, help="optional json file for the per-window arrays")
    a = ap.parse_args(argv)
    res = {Path(c).stem: per_window({"model": "neural", "checkpoint": c}, a.split) for c in a.checkpoints}
    print(f"per-lead ss ({a.split} random windows, n={len(next(iter(res.values())))})  leads {LEADS}")
    for k, v in res.items():
        print(f"  {k:12s} " + "  ".join(f"{x:+.3f}" for x in v.mean(0)))
    if a.ref:
        r = res[a.ref]
        for k, v in res.items():
            if k == a.ref:
                continue
            d = v - r; se = d.std(0, ddof=1) / np.sqrt(len(d))
            print(f"  {k} - {a.ref}: " + "  ".join(f"{m:+.4f}+-{2 * s:.4f}" for m, s in zip(d.mean(0), se)))
    if a.out:
        Path(a.out).write_text(json.dumps({k: v.tolist() for k, v in res.items()}))


if __name__ == "__main__":
    main()
