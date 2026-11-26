"""The simulation engine behind /simulate: applies the shock, runs the active model, cleans the output.

If the active model fails, a fallback model answers instead, so a window is never lost to an exception
(a failed window scores 0). Both paths are deterministic, so the null-shock test still holds.
"""
from __future__ import annotations

import logging
import time

import numpy as np

from .calendar import doy_from_sincos
from .config import MAX_STEPS
from .models import active_spec, build_model
from .models.base import TwinModel
from .physics import check_input, clamp_state, clean_output

log = logging.getLogger("twin")


class Engine:
    def __init__(self, spec: dict | None = None):
        try:
            self.fallback: TwinModel = build_model({"model": "damped_anomaly"})
        except Exception:                                   # artifacts missing: last resort
            self.fallback = build_model({"model": "persistence"})
        try:                                                # spec ARC-3: the server must always start
            self.spec = spec or active_spec()
            self.model: TwinModel = build_model(self.spec)
            self.model.warmup()
        except Exception as e:
            log.error("active model failed to load (%s: %s); serving fallback %s", type(e).__name__, e, self.fallback.name)
            self.spec, self.model = {"model": "fallback"}, self.fallback

    @property
    def name(self) -> str:
        return self.model.name

    def simulate(self, init_states, doy_sincos, steps: int, perturbation=None) -> np.ndarray:
        init = np.asarray(init_states, dtype="f4")
        pert = None if perturbation is None else np.asarray(perturbation, dtype="f4")
        check_input(init, pert)
        steps = int(steps)
        if not 1 <= steps <= MAX_STEPS:
            raise ValueError(f"steps must be in 1..{MAX_STEPS}")
        base = clamp_state(init.copy())                    # unshocked states (spec V2-CYC f)
        init = init.copy()
        if pert is not None:
            init[-1] = init[-1] + pert                      # the shock is a physical delta on the last input day
        init = clamp_state(init)
        doy = doy_from_sincos(doy_sincos)
        t0 = time.perf_counter()
        try:
            out = clean_output(self.model.rollout_shocked(init, base, doy, steps), steps)
            if not np.isfinite(out).all():
                raise FloatingPointError("non-finite rollout")
        except Exception as e:                              # never lose a window to a crash
            log.warning("model failed (%s: %s), using fallback %s", type(e).__name__, e, self.fallback.name)
            out = clean_output(self.fallback.rollout(init, doy, steps), steps)
        log.debug("rollout %d steps in %.1f ms", steps, (time.perf_counter() - t0) * 1000)
        return out
