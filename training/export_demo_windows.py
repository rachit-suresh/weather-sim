"""Export a few real 3-day input windows (held-out years) for the dashboard to send to the server.

These are INPUTS for the demo only (the dashboard still gets every forecast live from the server).
    python -m training.export_demo_windows
Writes dashboard/data/demo_windows.json.
"""
from __future__ import annotations

import json

import numpy as np

from .data import ROOT, TEST_YEARS, load
from .events import pick_events

ROUND = [2, 2, 2, 2, 0, 2]          # decimals per channel (t2m, tp, u10, v10, msl, tcwv)


def main():
    ds = load()
    out = []
    for name, t in pick_events(ds, TEST_YEARS).items():
        win = ds.data[t - 2:t + 1]
        states = [[np.round(win[d, c], ROUND[c]).tolist() for c in range(6)] for d in range(3)]
        doy = int(ds.doy[t])
        out.append({"name": name.replace("_", " "), "label": f"{name} (input ends {ds.ymd[t]})",
                    "doy": doy, "doy_sincos": [float(np.sin(2 * np.pi * doy / 365)), float(np.cos(2 * np.pi * doy / 365))],
                    "init_states": states})
    p = ROOT / "dashboard" / "data" / "demo_windows.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, separators=(",", ":")))
    print("wrote", p, [w["label"] for w in out], f"{p.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
