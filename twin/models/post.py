"""Post-processing wrapper (spec SHP-7): base model -> blend (ROL-3) -> variance rescale (SHP-3)
-> rain quantile map (SHP-4) -> clamp (done by the engine). Fitted by training/fit_post.py, stored in
artifacts/post/<name>.npz. Pure numpy and deterministic.

npz contents:
  blend_w   [L, 6]       weight of the damped-anomaly forecast per lead day and channel (0 = model only)
  scale     [L, 6]       anomaly multiplier per lead day and channel (1 = unchanged)
  scale_on  [6] bool     channels where the rescale is used (chosen on validation years)
  q_fc      [4, L, Q]    rain quantiles of the (blended, rescaled) forecast, per season and lead day
  q_true    [4, L, Q]    rain quantiles of the truth, same layout
  rainmap_on  [] bool    whether the rain map is used (chosen on validation years)
  blend_on    [] bool    whether the blend is used (spec 1.1 ROL-3; files without it: off)
Seasons by day of year d (0-based): 0 = winter (d < 59 or d >= 334), 1 = pre-monsoon (59 <= d < 151),
2 = monsoon (151 <= d < 273), 3 = post-monsoon (273 <= d < 334).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..climatology import clim_sequence
from ..config import ARTIFACTS, TP
from .base import TwinModel


def season_of(doy: float) -> int:
    d = doy % 365
    if d < 59 or d >= 334:
        return 0
    if d < 151:
        return 1
    if d < 273:
        return 2
    return 3


def _lead(arr: np.ndarray, steps: int) -> np.ndarray:
    return arr[np.minimum(np.arange(steps), arr.shape[0] - 1)]


class PostProcessed(TwinModel):
    def __init__(self, base: TwinModel, file: str | Path, blend=None, rescale=None, rainmap=None):
        """blend/rescale/rainmap: None = use the choice stored in the file, True/False = force."""
        from .baselines import DampedAnomaly
        p = Path(file)
        z = np.load(p if p.is_absolute() else ARTIFACTS / p)
        self.base, self.damped = base, DampedAnomaly()
        self.blend_w = z["blend_w"].astype("f4")
        self.scale = z["scale"].astype("f4")
        self.scale_on = z["scale_on"].astype(bool)
        self.q_fc, self.q_true = z["q_fc"].astype("f4"), z["q_true"].astype("f4")
        self.use_blend = bool(z["blend_on"]) if (blend is None and "blend_on" in z.files) else bool(blend)
        self.use_rescale = True if rescale is None else bool(rescale)
        self.use_rainmap = bool(z["rainmap_on"]) if rainmap is None else bool(rainmap)
        tags = [t for t, on in (("blend", self.use_blend), ("rescale", self.use_rescale), ("rainmap", self.use_rainmap)) if on]
        self.name = f"{base.name}+{'+'.join(tags) or 'none'}"

    def apply(self, out: np.ndarray, init_states: np.ndarray, doy: float) -> np.ndarray:
        steps = out.shape[0]
        out = out.astype("f4").copy()
        clim = clim_sequence(doy, steps)
        if self.use_blend:
            w = _lead(self.blend_w, steps)[:, :, None, None]
            out = (1 - w) * out + w * self.damped.rollout(init_states, doy, steps)
        if self.use_rescale:
            s = np.where(self.scale_on[None, :], _lead(self.scale, steps), 1.0)[:, :, None, None]
            out = clim + (out - clim) * s
        if self.use_rainmap:
            for k in range(steps):
                si, li = season_of(doy + k + 1), min(k, self.q_fc.shape[1] - 1)
                out[k, TP] = np.interp(np.maximum(out[k, TP], 0.0), self.q_fc[si, li], self.q_true[si, li])
        return out

    def rollout(self, init_states, doy, steps):
        return self.apply(self.base.rollout(init_states, doy, steps), init_states, doy)

    def warmup(self):
        self.base.warmup()
