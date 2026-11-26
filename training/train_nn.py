"""Train any torch network family as an autoregressive 1-day stepper in normalised-anomaly space.

    python -m training.train_nn --family cnn  --name cnn_base
    python -m training.train_nn --family unet --name unet_base --net '{"width": 48}'
    python -m training.train_nn --family cnn  --name cnn_sqrt --rain sqrt --curriculum 1:4,2:4,4:6 --var-loss 0.1

Unroll curriculum "1:4,2:4,4:6" = 4 epochs with 1-step loss, then 4 epochs unrolling 2 steps, then 6 with 4.
Saves artifacts/models/<name>.pt (state_dict + config only) and scores it with training.evaluate.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from twin.calendar import clim_index, sincos
from twin.climatology import load_static
from twin.codec import Codec, clim_feature
from twin.config import ARTIFACTS, LAT
from twin.nets import build_net
from twin.nets.common import edge_distance, static_features
from twin.nets.rollout import rollout

from .data import TRAIN_YEARS, VAL_YEARS, load, window_starts


class Tensors:
    """All days pre-encoded once: anomalies Z, target-day climatology features, sin/cos."""

    def __init__(self, codec: Codec, device):
        ds = load()
        self.ds = ds
        doys = ds.doy.astype(float)
        self.Z = torch.from_numpy(np.concatenate([codec.encode(ds.data[i:i + 512], doys[i:i + 512])
                                                  for i in range(0, len(doys), 512)])).to(device)
        table = clim_feature(list(range(366)))                         # index by clim slot
        ci = np.array([clim_index(d) for d in doys])
        self.CF = torch.from_numpy(table[ci]).to(device)
        self.SC = torch.from_numpy(np.stack([sincos(d) for d in doys])).to(device)
        self.fixed = torch.from_numpy(np.concatenate([static_features(load_static()), edge_distance()])).to(device)

    def batch(self, t: np.ndarray, k: int):
        t = torch.as_tensor(t, device=self.Z.device)
        hist = torch.stack([self.Z[t - 2], self.Z[t - 1], self.Z[t]], dim=1)
        fut = torch.stack([t + j for j in range(1, k + 1)], dim=1)
        return hist, self.CF[fut], self.SC[fut], self.Z[fut]


def anomaly_std(rain_mode: str) -> list[float]:
    ds = load(); m = ds.mask(TRAIN_YEARS)
    c = Codec(rain_mode=rain_mode)
    z = np.concatenate([c.encode(ds.data[i:i + 512], ds.doy[i:i + 512].astype(float))
                        for i in np.where(m)[0][::512]])                  # anomaly with std 1 -> raw anomaly
    return [float(x) for x in z.std(axis=(0, 2, 3))]


def loss_fn(pred, true, lat_w, ch_w, var_w: float):
    se = ((pred - true) ** 2 * lat_w).mean(dim=(0, 1, 3, 4))           # per channel
    loss = (se * ch_w).sum() / ch_w.sum()
    if var_w > 0:                                                      # keep spatial spread (anti-blur)
        sp, st = pred.std(dim=(-2, -1)), true.std(dim=(-2, -1))
        loss = loss + var_w * ((sp - st).abs().mean(dim=(0, 1)) * ch_w).sum() / ch_w.sum()
    return loss


def parse_curriculum(s: str) -> list[tuple[int, int]]:
    return [(int(a), int(b)) for a, b in (p.split(":") for p in s.split(","))]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--net", default="{}", help="json kwargs for the network")
    ap.add_argument("--rain", default="none", choices=["none", "sqrt"])
    ap.add_argument("--curriculum", default="1:6,2:4,4:4")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--var-loss", type=float, default=0.0)
    ap.add_argument("--channel-weights", default="1,1,1,1,1,1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--no-eval", action="store_true")
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed); np.random.seed(a.seed); torch.set_num_threads(a.threads)
    dev = torch.device(a.device)
    codec = Codec(rain_mode=a.rain)
    codec.anom_std = anomaly_std(a.rain)
    T = Tensors(codec, dev)
    net_cfg = json.loads(a.net)
    net = build_net(a.family, **net_cfg).to(dev)
    n_par = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    cur = parse_curriculum(a.curriculum)
    total_steps = sum(n_ep * -(-len(window_starts(T.ds, TRAIN_YEARS, u)) // a.batch) for u, n_ep in cur)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total_steps + 1, pct_start=0.05)
    lat_w = torch.tensor(np.cos(np.deg2rad(LAT)) / np.cos(np.deg2rad(LAT)).mean(), dtype=torch.float32, device=dev)[:, None]
    ch_w = torch.tensor([float(x) for x in a.channel_weights.split(",")], device=dev)
    print(f"{a.family} {net_cfg} params={n_par:,} device={dev} rain={a.rain} curriculum={cur}")

    best, out_path = float("inf"), ARTIFACTS / "models" / f"{a.name}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    step_count, ep = 0, 0
    for unroll, n_ep in cur:
        tr_idx = window_starts(T.ds, TRAIN_YEARS, unroll)
        va_idx = window_starts(T.ds, VAL_YEARS, unroll, stride=3)
        for _ in range(n_ep):
            ep += 1; t0 = time.time(); net.train(); tot = 0.0
            perm = np.random.permutation(tr_idx)
            for i in range(0, len(perm), a.batch):
                hist, cf, sc, true = T.batch(perm[i:i + a.batch], unroll)
                pred = rollout(net, hist, cf, sc, T.fixed, unroll)
                loss = loss_fn(pred, true, lat_w, ch_w, a.var_loss)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                if step_count < total_steps:
                    sched.step(); step_count += 1
                tot += loss.item() * len(hist)
            net.eval(); vl = 0.0
            with torch.no_grad():
                for i in range(0, len(va_idx), 128):
                    hist, cf, sc, true = T.batch(va_idx[i:i + 128], unroll)
                    vl += loss_fn(rollout(net, hist, cf, sc, T.fixed, unroll), true, lat_w, ch_w, 0.0).item() * len(hist)
            vl /= len(va_idx)
            print(f"ep {ep:3d} unroll {unroll}  train {tot / len(tr_idx):.4f}  val {vl:.4f}  {time.time() - t0:.0f}s", flush=True)
            if unroll == cur[-1][0] and vl < best:
                best = vl
                torch.save({"family": a.family, "net_config": net_cfg, "codec": codec.to_dict(),
                            "state_dict": {k: v.cpu() for k, v in net.state_dict().items()},
                            "train_args": vars(a)}, out_path)
    print("saved", out_path, "best val", round(best, 4))
    if not a.no_eval:
        from .evaluate import evaluate
        res = evaluate({"model": "neural", "checkpoint": str(out_path.relative_to(ARTIFACTS))}, "val", a.name)
        print(json.dumps({k: res[k] for k in ("auto_events_of_70", "auto_random_of_70", "gate")}),
              "SS", round(res["random"]["mean_ss"], 3), "CSI", round(res["random"]["mean_csi_skill"], 3),
              "blur", res["random"]["blur_ratio"], "causal", round(res["shocks"]["causal_pts"], 2))


if __name__ == "__main__":
    main()
