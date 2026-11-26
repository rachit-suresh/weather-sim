"""Family 2: LightGBM per cell, rolled out autoregressively in normalised-anomaly space.

Artifacts in artifacts/<dir>/: meta.json (codec) + model_<channel>.txt (one booster per channel).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from ..calendar import sincos
from ..climatology import load_static
from ..codec import Codec, clim_feature
from ..config import ARTIFACTS, CHANNELS, NLAT, NLON
from ..lgbm_features import features
from .base import TwinModel

Z_CLAMP = 12.0


class LGBMModel(TwinModel):
    def __init__(self, directory: str | Path):
        import lightgbm as lgb
        d = Path(directory)
        if not d.is_absolute():
            d = ARTIFACTS / d
        meta = json.loads((d / "meta.json").read_text())
        self.codec = Codec.from_dict(meta["codec"])
        self.boosters = [lgb.Booster(model_file=str(d / f"model_{c}.txt")) for c in CHANNELS]
        self.threads = int(os.environ.get("TWIN_THREADS", "4"))
        st = load_static()
        from ..nets.common import edge_distance
        self.fixed = np.concatenate([np.stack([st[0], st[1] / 1000.0]), edge_distance()]).astype("f4")
        self.name = f"lgbm:{d.name}"

    def rollout(self, init_states, doy, steps):
        out_doys = [doy + k for k in range(1, steps + 1)]
        hist = self.codec.encode(init_states, [doy - 2, doy - 1, doy])[None]           # [1,3,6,H,W]
        cf = clim_feature(out_doys)
        zs = []
        for k in range(steps):
            X = features(hist, cf[k:k + 1], sincos(out_doys[k])[None], self.fixed)
            dz = np.stack([b.predict(X, num_threads=self.threads) for b in self.boosters], axis=1)  # [HW, 6]
            z = np.clip(hist[0, 2] + dz.T.reshape(6, NLAT, NLON), -Z_CLAMP, Z_CLAMP).astype("f4")
            zs.append(z)
            hist = np.concatenate([hist[:, 1:], z[None, None]], axis=1)
        return self.codec.decode(np.stack(zs), out_doys)
