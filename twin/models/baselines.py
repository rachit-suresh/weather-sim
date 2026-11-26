"""Family 1: classical baselines. No neural network; also used as fallbacks and as priors for learned models."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..climatology import clim_at, clim_sequence
from ..config import ARTIFACTS
from .base import TwinModel


class Persistence(TwinModel):
    """Tomorrow = today. Reacts to shocks but never spreads them."""

    name = "persistence"

    def rollout(self, init_states, doy, steps):
        return np.repeat(init_states[-1][None], steps, axis=0)


class Climatology(TwinModel):
    """Long-term mean for each calendar day. Ignores the input (so it ignores shocks too)."""

    name = "climatology"

    def rollout(self, init_states, doy, steps):
        return clim_sequence(doy, steps)


class DampedAnomaly(TwinModel):
    """forecast(lead) = clim(lead) + a[lead, channel] * (today - clim(today)).

    `a` is fitted by least squares on the training years (training/fit_damped.py) and stored in
    artifacts/damped.json. Local by construction: a shock stays exactly where it was applied.
    """

    name = "damped_anomaly"

    def __init__(self, path: Path | None = None):
        js = json.loads((path or ARTIFACTS / "damped.json").read_text())
        self.alpha = np.asarray(js["alpha"], "f4")          # [max_lead, 6]

    def rollout(self, init_states, doy, steps):
        anom = init_states[-1] - clim_at(doy)
        clim = clim_sequence(doy, steps)
        a = self.alpha[np.minimum(np.arange(steps), len(self.alpha) - 1)]   # [steps, 6]
        return clim + a[:, :, None, None] * anom[None]
