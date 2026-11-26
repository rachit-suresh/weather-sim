"""Model interface. Every family (baselines, CNN, U-Net, hybrid, ensembles) implements `TwinModel`."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class TwinModel(ABC):
    """A deterministic forecaster in PHYSICAL units.

    `rollout` receives the 3 input days with any shock already added to the last day, and must
    return [steps, 6, 32, 36]. It must be a pure function of its inputs (no state kept between
    calls, no unseeded randomness), otherwise the judge's null-shock test fails.
    """

    name: str = "base"

    @abstractmethod
    def rollout(self, init_states: np.ndarray, doy: float, steps: int) -> np.ndarray:
        ...

    def rollout_shocked(self, init_states: np.ndarray, base_states: np.ndarray, doy: float, steps: int) -> np.ndarray:
        """`init_states` with the shock added, `base_states` = the same request before the shock (spec V2-CYC f).
        Models whose behaviour depends on the unshocked states (cyclone mode) override this; others ignore it."""
        return self.rollout(init_states, doy, steps)

    def warmup(self) -> None:
        """Optional: run once at start-up (JIT, thread pools) so scored calls are fast."""
