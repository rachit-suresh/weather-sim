"""Contract tests that run without a server:  python -m pytest tests -q   (or python tests/test_engine.py)

Checks: output shape and validity, zero shock == no shock (judge's null test), determinism across calls,
a real shock changes the output near its box, and a 10-step rollout is far under the 5 s limit.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from twin.climatology import clim_sequence          # noqa: E402
from twin.config import load_stats                  # noqa: E402
from twin.engine import Engine                      # noqa: E402

DOY = 160.0
SINCOS = [float(np.sin(2 * np.pi * DOY / 365)), float(np.cos(2 * np.pi * DOY / 365))]


def _init():
    return np.ascontiguousarray(clim_sequence(DOY - 3, 3))      # 3 plausible days


def _box(lat0, lat1, lon0, lon1):
    lat = np.arange(6.5, 38, 1.0); lon = np.arange(65.5, 101, 1.0)
    return ((lat >= lat0) & (lat <= lat1))[:, None] & ((lon >= lon0) & (lon <= lon1))[None, :]


def test_contract():
    eng = Engine()
    init = _init(); _, std = load_stats()
    t0 = time.perf_counter()
    base = eng.simulate(init, SINCOS, 10)
    assert time.perf_counter() - t0 < 2.0
    assert base.shape == (10, 6, 32, 36) and np.isfinite(base).all()
    assert base[:, 1].min() >= 0 and 233.15 <= base[:, 0].min() and base[:, 0].max() <= 333.15
    again = eng.simulate(init, SINCOS, 10)
    zero = eng.simulate(init, SINCOS, 10, np.zeros((6, 32, 36), "f4"))
    assert (np.abs(base - again) / std[None, :, None, None]).max() <= 1e-3
    assert (np.abs(base - zero) / std[None, :, None, None]).max() <= 1e-3
    pert = np.zeros((6, 32, 36), "f4"); m = _box(8, 20, 82, 94); pert[0][m] = 3.0
    shocked = eng.simulate(init, SINCOS, 10, pert)
    d = np.abs(shocked - base)[0, 0]
    assert d[m].mean() > d[~m].mean()


if __name__ == "__main__":
    test_contract(); print("ok")
