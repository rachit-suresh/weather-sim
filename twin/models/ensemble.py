"""Ensembles: deterministic weighted blends of member models, with weights per lead and channel.

Averaging lowers RMSE but also lowers spatial variance (blur). `restore_variance` rescales the blended
anomaly so its spatial spread matches the mean spread of the members (see docs/MODEL_FAMILIES.md).
"""
from __future__ import annotations

import numpy as np

from ..climatology import clim_sequence
from .base import TwinModel


class BlendEnsemble(TwinModel):
    def __init__(self, members: list[TwinModel], weights=None, restore_variance: bool = False):
        self.members = members
        n = len(members)
        # weights: None (equal), [n], or [n, max_lead, 6]
        w = np.ones(n, "f4") / n if weights is None else np.asarray(weights, "f4")
        self.w = w
        self.restore_variance = restore_variance
        self.name = "ensemble[" + ",".join(m.name for m in members) + "]"

    def _weights(self, steps):
        w = self.w
        if w.ndim == 1:
            w = np.broadcast_to(w[:, None, None], (len(self.members), steps, 6))
        else:
            w = w[:, np.minimum(np.arange(steps), w.shape[1] - 1)]
        return w / w.sum(axis=0, keepdims=True)

    def rollout(self, init_states, doy, steps):
        outs = np.stack([m.rollout(init_states, doy, steps) for m in self.members])   # [n, steps, 6, H, W]
        w = self._weights(steps)[..., None, None]
        blend = (w * outs).sum(axis=0)
        if self.restore_variance:
            clim = clim_sequence(doy, steps)
            a_b = blend - clim
            sd_b = a_b.std(axis=(-2, -1), keepdims=True)
            sd_m = (w * (outs - clim[None]).std(axis=(-2, -1), keepdims=True)).sum(axis=0)
            blend = clim + a_b * np.where(sd_b > 1e-6, sd_m / np.maximum(sd_b, 1e-6), 1.0)
        return blend

    def warmup(self):
        for m in self.members:
            m.warmup()
