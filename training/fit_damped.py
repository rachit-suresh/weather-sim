"""Family 1 baseline: fit damped anomaly persistence by least squares on the training years.

alpha[lead, channel] = sum(a_t * a_{t+lead}) / sum(a_t^2),  a = state - climatology
Writes artifacts/damped.json.     python -m training.fit_damped
"""
from __future__ import annotations

import json

import numpy as np

from twin.calendar import clim_index
from twin.climatology import load_climatology
from twin.config import ARTIFACTS, MAX_STEPS

from .data import TRAIN_YEARS, load, window_starts


def fit(max_lead: int = MAX_STEPS) -> np.ndarray:
    ds = load(); clim = load_climatology()
    anom = ds.data - clim[[clim_index(d) for d in ds.doy]]
    alpha = np.zeros((max_lead, 6), "f4")
    for L in range(1, max_lead + 1):
        t = window_starts(ds, TRAIN_YEARS, L)
        x, y = anom[t], anom[t + L]
        alpha[L - 1] = (x * y).sum(axis=(0, 2, 3)) / (x * x).sum(axis=(0, 2, 3))
    return np.clip(alpha, 0.0, 1.0)


def main():
    alpha = fit()
    (ARTIFACTS / "damped.json").write_text(json.dumps({"alpha": alpha.round(5).tolist()}, indent=1))
    print("alpha lead1", alpha[0].round(3), "lead8", alpha[7].round(3))


if __name__ == "__main__":
    main()
