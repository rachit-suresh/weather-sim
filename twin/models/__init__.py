"""Model registry. The active model is described by artifacts/active.json (or env TWIN_MODEL_SPEC = path to a json).

Spec examples:
  {"model": "persistence"}
  {"model": "damped_anomaly"}
  {"model": "neural", "checkpoint": "models/cnn_base.pt"}
  {"model": "lgbm", "dir": "lgbm/lgbm_base"}
  {"model": "ensemble", "members": [{...}, {...}], "weights": [0.5, 0.5], "restore_variance": true}
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from ..config import ARTIFACTS
from .base import TwinModel


def build_model(spec: dict) -> TwinModel:
    kind = spec["model"]
    if kind == "persistence":
        from .baselines import Persistence
        return Persistence()
    if kind == "climatology":
        from .baselines import Climatology
        return Climatology()
    if kind == "damped_anomaly":
        from .baselines import DampedAnomaly
        return DampedAnomaly()
    if kind == "neural":
        from .neural import NeuralModel
        return NeuralModel(spec["checkpoint"])
    if kind == "lgbm":
        from .lgbm import LGBMModel
        return LGBMModel(spec["dir"])
    if kind == "ensemble":
        from .ensemble import BlendEnsemble
        return BlendEnsemble([build_model(m) for m in spec["members"]], spec.get("weights"),
                             spec.get("restore_variance", False))
    raise ValueError(f"unknown model kind {kind!r}")


def active_spec() -> dict:
    p = os.environ.get("TWIN_MODEL_SPEC")
    path = Path(p) if p else ARTIFACTS / "active.json"
    return json.loads(path.read_text())
