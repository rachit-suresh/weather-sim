"""Network families. Add a family: write the module, register it in FAMILIES, train it with
`python -m training.train_nn --family <name>`. See docs/MODEL_FAMILIES.md for the full list and plan.

Families still to add: convgru, gnn, windowed_vit, hybrid_advection (and lightgbm, which is not a torch net).
"""
from __future__ import annotations

from .common import N_IN, N_OUT


def build_net(family: str, **cfg):
    if family == "cnn":
        from .cnn import LocalCNN
        return LocalCNN(N_IN, N_OUT, **cfg)
    if family == "unet":
        from .unet import SmallUNet
        return SmallUNet(N_IN, N_OUT, **cfg)
    raise ValueError(f"unknown family {family!r}")
