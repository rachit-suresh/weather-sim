"""Any torch network family wrapped as a TwinModel. Checkpoint = {family, net_config, codec, state_dict}."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..calendar import sincos
from ..climatology import load_static
from ..codec import Codec, clim_feature
from ..config import ARTIFACTS
from .base import TwinModel


class NeuralModel(TwinModel):
    def __init__(self, checkpoint: str | Path):
        import torch
        torch.set_num_threads(int(os.environ.get("TWIN_THREADS", "4")))     # sandbox has 4 cores; never oversubscribe
        torch.set_grad_enabled(False)
        from ..nets import build_net
        from ..nets.common import edge_distance, static_features

        path = Path(checkpoint)
        if not path.is_absolute():
            path = ARTIFACTS / path
        ck = torch.load(path, map_location="cpu", weights_only=True)
        self.torch = torch
        self.name = f"{ck['family']}:{path.stem}"
        self.codec = Codec.from_dict(ck["codec"])
        self.net = build_net(ck["family"], **ck["net_config"]).eval()
        self.net.load_state_dict(ck["state_dict"])
        self.fixed = torch.from_numpy(np.concatenate([static_features(load_static()), edge_distance()]))

    def rollout(self, init_states, doy, steps):
        torch = self.torch
        from ..nets.rollout import rollout
        out_doys = [doy + k for k in range(1, steps + 1)]
        hist = torch.from_numpy(self.codec.encode(init_states, [doy - 2, doy - 1, doy]))[None]
        clim_n = torch.from_numpy(clim_feature(out_doys))[None]
        sc = torch.from_numpy(np.stack([sincos(d) for d in out_doys]))[None]
        z = rollout(self.net, hist, clim_n, sc, self.fixed, steps)[0].numpy()
        return self.codec.decode(z, out_doys)

    def warmup(self):
        x = np.repeat(self.codec.inv(self.codec.clim_t([100.0])), 3, axis=0)
        self.rollout(x, 100.0, 2)
