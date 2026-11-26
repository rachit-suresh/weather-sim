"""Network families. Add a family: write the module, register it in FAMILIES, train it with
`python -m training.train_nn --family <name>`. See docs/MODEL_FAMILIES.md for the full list.

LightGBM (family 2) is not a torch net: see twin/models/lgbm.py and training/train_lgbm.py.
"""
from __future__ import annotations

from .common import N_IN, N_OUT

FAMILIES = {
    "cnn": ("cnn", "LocalCNN"),                     # 3  local ResNet CNN
    "unet": ("unet", "SmallUNet"),                  # 4  one-level U-Net
    "convgru": ("convgru", "ConvGRUNet"),           # 5  CNN + recurrent memory (time series)
    "gnn": ("gnn", "GridGNN"),                      # 6  message passing on the grid graph
    "hybrid": ("attention", "HybridCNNTransformer"),  # 7a CNN + window-attention transformer
    "vit": ("attention", "WindowViT"),              # 7b pure window transformer on 2x2 patches
    "fno": ("fno", "FNO"),                          # 8  Fourier neural operator (control: global mixing)
    "physics": ("physics", "AdvectionHybrid"),      # 10 learned advection + sources
}


def build_net(family: str, **cfg):
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; known: {sorted(FAMILIES)}")
    import importlib
    mod, cls = FAMILIES[family]
    return getattr(importlib.import_module(f".{mod}", __name__), cls)(N_IN, N_OUT, **cfg)
