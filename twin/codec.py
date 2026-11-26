"""Physical state <-> normalised climatology anomaly. Shared by training and inference so they never drift apart.

z = (f(x) - f(clim(doy))) / anom_std          f = identity, or sqrt for rain when rain_mode == "sqrt"
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .climatology import load_climatology
from .calendar import clim_index
from .config import TP, load_stats


def clim_feature(doys) -> np.ndarray:
    """Network input: climatology of the given days scaled by the global stats.json -> [len, 6, H, W]."""
    mean, std = load_stats()
    c = load_climatology()
    return ((np.stack([c[clim_index(d)] for d in doys]) - mean[:, None, None]) / std[:, None, None]).astype("f4")


@dataclass
class Codec:
    anom_std: list = field(default_factory=lambda: [1.0] * 6)   # per channel, in transformed units
    rain_mode: str = "none"                                     # "none" | "sqrt"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Codec":
        return cls(**d)

    # --- transforms on [..., 6, H, W] arrays ---
    def fwd(self, x: np.ndarray) -> np.ndarray:
        x = np.array(x, dtype="f4", copy=True)
        if self.rain_mode == "sqrt":
            x[..., TP, :, :] = np.sqrt(np.maximum(x[..., TP, :, :], 0.0))
        return x

    def inv(self, x: np.ndarray) -> np.ndarray:
        x = np.array(x, dtype="f4", copy=True)
        if self.rain_mode == "sqrt":
            x[..., TP, :, :] = np.maximum(x[..., TP, :, :], 0.0) ** 2
        return x

    def clim_t(self, doys) -> np.ndarray:
        """Transformed climatology for each day in `doys` -> [len, 6, H, W]."""
        c = load_climatology()
        return self.fwd(np.stack([c[clim_index(d)] for d in doys]))

    def encode(self, x: np.ndarray, doys) -> np.ndarray:
        s = np.asarray(self.anom_std, "f4")[:, None, None]
        return (self.fwd(x) - self.clim_t(doys)) / s

    def decode(self, z: np.ndarray, doys) -> np.ndarray:
        s = np.asarray(self.anom_std, "f4")[:, None, None]
        return self.inv(z * s + self.clim_t(doys))
