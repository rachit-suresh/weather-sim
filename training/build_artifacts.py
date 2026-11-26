"""Build the small artifacts the server ships with:

  artifacts/climatology.npz   clim [366, 6, 32, 36] (15-day smoothed, one year only) + static [2, 32, 36]
  artifacts/stats.json        per-channel mean/std (same as the kit's stats.json)

    python -m training.build_artifacts
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from twin.config import ARTIFACTS, CHANNELS

from .data import load


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2005-2021", help="year range for the climatology, e.g. 2005-2021")
    a = ap.parse_args(argv)
    from monsoon_r1.stats import climatology            # the kit's own builder, so we match the grader's reference

    ds = load()
    y0, y1 = (int(x) for x in a.years.split("-"))
    m = (ds.years >= y0) & (ds.years <= y1)
    clim = climatology(ds.data[m], ds.ymd[m], smooth_days=15)
    ARTIFACTS.mkdir(exist_ok=True)
    np.savez_compressed(ARTIFACTS / "climatology.npz", clim=clim.astype("f4"), static=ds.static.astype("f4"))
    st = {"channels": list(CHANNELS), "mean": [float(x) for x in ds.data.mean(axis=(0, 2, 3), dtype="f8")],
          "std": [float(x) for x in ds.data.std(axis=(0, 2, 3), dtype="f8")]}
    (ARTIFACTS / "stats.json").write_text(json.dumps(st, indent=1))
    print("wrote", ARTIFACTS / "climatology.npz", clim.shape, "and stats.json")


if __name__ == "__main__":
    main()
