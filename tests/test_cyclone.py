"""Cyclone mode (spec V2-CYC):  python -m pytest tests/test_cyclone.py -q

- no detection -> output is the base model bit for bit (direct and through the engine);
- a synthetic deep low over the Bay is detected; zero shock == no shock exactly; the mask does not depend on the
  shock; outside the mask (and, for variant A, on t2m / tp / tcwv everywhere) the output is the base model's;
- the mask is 1 up to R cells and 0 from R + T cells.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from twin.climatology import clim_sequence           # noqa: E402
from twin.config import LAT, LON, MSL, TCWV, TP, T2M  # noqa: E402
from twin.engine import Engine                       # noqa: E402
from twin.models.cyclone import RADIUS, TAPER, CycloneMode, circle, detect  # noqa: E402
from twin.models.neural import NeuralModel           # noqa: E402

DOY = 300.0
SINCOS = [float(np.sin(2 * np.pi * DOY / 365)), float(np.cos(2 * np.pi * DOY / 365))]
BASE, MOVER = "models/r3_cnn.pt", "models/v2_e7c.pt"


def _spec(variant):
    return {"model": "cyclone_mode", "base": BASE, "mover": MOVER, "variant": variant}


def _calm():
    return np.ascontiguousarray(clim_sequence(DOY - 3, 3))


def _low(depth_pa=2000.0, at=(15.5, 87.5)):
    x = _calm().copy()
    ii, jj = np.indices((len(LAT), len(LON)))
    i, j = int(np.argmin(np.abs(LAT - at[0]))), int(np.argmin(np.abs(LON - at[1])))
    x[:, MSL] -= depth_pa * np.exp(-((ii - i) ** 2 + (jj - j) ** 2) / (2 * 2.0 ** 2))
    return x, (i, j)


def test_mask_shape():
    m = circle((16, 18))
    ii, jj = np.indices(m.shape); d = np.hypot(ii - 16, jj - 18)
    assert (m[d <= RADIUS] == 1).all() and (m[d >= RADIUS + TAPER] == 0).all()
    assert ((m >= 0) & (m <= 1)).all()


def test_no_detection_is_base_bit_for_bit():
    init = _calm()
    assert detect(init[-1], DOY) is None
    base = NeuralModel(BASE)
    for v in ("A", "B"):
        cyc = CycloneMode(BASE, MOVER, v)
        assert np.array_equal(cyc.rollout(init, DOY, 10), base.rollout(init, DOY, 10))
    e_cyc, e_base = Engine(_spec("A")), Engine({"model": "neural", "checkpoint": BASE})
    assert e_cyc.name.startswith("cyclone_mode")
    pert = np.zeros((6, 32, 36), "f4"); pert[0, 5:15, 18:30] = 3.0
    assert np.array_equal(e_cyc.simulate(init, SINCOS, 10), e_base.simulate(init, SINCOS, 10))
    assert np.array_equal(e_cyc.simulate(init, SINCOS, 10, pert), e_base.simulate(init, SINCOS, 10, pert))
    deep = np.zeros((6, 32, 36), "f4"); deep[MSL, 8:12, 20:24] = -3000.0     # the SHOCK makes a deep low:
    assert np.array_equal(e_cyc.simulate(init, SINCOS, 10, deep), e_base.simulate(init, SINCOS, 10, deep))  # ignored


def test_detected_low_null_exact_and_mask_fixed():
    init, (i, j) = _low()
    assert detect(init[-1], DOY) == (i, j)
    base = Engine({"model": "neural", "checkpoint": BASE})
    pert = np.zeros((6, 32, 36), "f4"); pert[0, 5:15, 18:30] = 3.0; pert[5, 5:15, 18:30] = 5.0
    for v in ("A", "B"):
        eng = Engine(_spec(v))
        cyc = eng.model
        none = eng.simulate(init, SINCOS, 10)
        zero = eng.simulate(init, SINCOS, 10, np.zeros((6, 32, 36), "f4"))
        assert np.array_equal(none, zero)                                   # judge's null test, exactly
        shocked = eng.simulate(init, SINCOS, 10, pert)
        m, _ = cyc.masks(init, DOY, 10)
        assert m is not None and m[0, i, j] == 1.0
        r3 = base.simulate(init, SINCOS, 10, pert)
        out = m == 0
        assert np.array_equal(shocked.transpose(1, 0, 2, 3)[:, out], r3.transpose(1, 0, 2, 3)[:, out])
        if v == "A":
            for c in (T2M, TP, TCWV):
                assert np.array_equal(shocked[:, c], r3[:, c])
        assert not np.array_equal(shocked[:, MSL], r3[:, MSL])


if __name__ == "__main__":
    test_mask_shape(); test_no_detection_is_base_bit_for_bit(); test_detected_low_null_exact_and_mask_fixed(); print("ok")
