"""Train any torch network family as an autoregressive 1-day stepper in normalised-anomaly space.

    python -m training.train_nn --family cnn --name cnn_x                       # spec v1 recipe (defaults)
    python -m training.train_nn --family unet --name u_edge --edge-relax        # + F4 (spec EDG-2)
    python -m training.train_nn --family unet --name u_ll --latlon              # + F6 (spec EDG-2)
    python -m training.train_nn --family unet --name u_d4 --rain-head two_part  # + D4 (spec RAIN-3)
    python -m training.train_nn --family unet --name u_sg --shock-loss 5        # + shock guard (spec SHP-8)
    python -m training.train_nn --family cnn --name c_e6 --net 'pad="zeros"' --random-crop   # spec V2-E6

Unroll curriculum "1:4,2:3,4:3,8:2" = 4 epochs with 1-step loss, then 3 epochs unrolling 2 steps, and so on.
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
from twin.config import ARTIFACTS, LAT, NLAT, NLON, T2M, TCWV, TP, load_stats
from twin.nets import build_net
from twin.nets.common import EDGE_RELAX_DEFAULT, edge_distance, fixed_maps, n_inputs
from twin.nets.rollout import rollout

from .data import ROOT, TRAIN_YEARS, VAL_YEARS, load, window_starts

HEAVY_MM = 10.0     # RAIN-2 threshold (same as the judge's heavy-rain CSI)
WET_MM = 1.0        # RAIN-3 "it rains" threshold for the two-part head
SHOCK_BOX = 12      # SHP-8 random shock box size (cells)
SHOCK_LIMIT = 4.0   # SHP-8 penalty starts above 4 std (judge fails above 5)
CROP_MIN = (24, 28) # V2-E6 smallest random training crop (rows, cols); the largest is the full 32 x 36 grid


class Tensors:
    """All days pre-encoded once: anomalies Z, target-day climatology features, sin/cos, masks."""

    def __init__(self, codec: Codec, device, latlon: bool = False):
        ds = load()
        self.ds = ds
        doys = ds.doy.astype(float)
        self.Z = torch.from_numpy(np.concatenate([codec.encode(ds.data[i:i + 512], doys[i:i + 512])
                                                  for i in range(0, len(doys), 512)])).to(device)
        table = clim_feature(list(range(366)))                         # index by clim slot
        ci = np.array([clim_index(d) for d in doys])
        self.CF = torch.from_numpy(table[ci]).to(device)
        self.SC = torch.from_numpy(np.stack([sincos(d) for d in doys])).to(device)
        self.fixed = torch.from_numpy(fixed_maps(load_static(), latlon)).to(device)
        self.heavy = torch.from_numpy(ds.data[:, TP] > HEAVY_MM).to(device)          # RAIN-2
        self.wet = torch.from_numpy(ds.data[:, TP] > WET_MM).to(device)              # RAIN-3

    def batch(self, t: np.ndarray, k: int):
        t = torch.as_tensor(t, device=self.Z.device)
        hist = torch.stack([self.Z[t - 2], self.Z[t - 1], self.Z[t]], dim=1)
        fut = torch.stack([t + j for j in range(1, k + 1)], dim=1)
        return hist, self.CF[fut], self.SC[fut], self.Z[fut], self.heavy[fut], self.wet[fut]


def anomaly_std(rain_mode: str) -> list[float]:
    ds = load(); m = ds.mask(TRAIN_YEARS)
    c = Codec(rain_mode=rain_mode)
    z = np.concatenate([c.encode(ds.data[i:i + 512], ds.doy[i:i + 512].astype(float))
                        for i in np.where(m)[0][::512]])                  # anomaly with std 1 -> raw anomaly
    return [float(x) for x in z.std(axis=(0, 2, 3))]


def random_crop(tensors, fixed: torch.Tensor, lat_w: torch.Tensor):
    """V2-E6 (b): cut every spatial tensor of one batch to the same random sub-window (height 24-32, width 28-36,
    random position). tensors = [..., H, W] maps; fixed = [lsm, oro, edge (, lat, lon)] with the edge-distance map
    recomputed for the crop, so the net sees 'edge' behaviour at varying places; lat_w [H, 1] latitude weights."""
    h = np.random.randint(CROP_MIN[0], NLAT + 1); w = np.random.randint(CROP_MIN[1], NLON + 1)
    i0 = np.random.randint(0, NLAT - h + 1); j0 = np.random.randint(0, NLON - w + 1)
    rows, cols = slice(i0, i0 + h), slice(j0, j0 + w)
    fx = fixed[:, rows, cols].clone()
    fx[2] = torch.from_numpy(edge_distance(h, w)[0]).to(fx)
    return [x[..., rows, cols] for x in tensors], fx, lat_w[rows]


def loss_fn(pred, true, lat_w, ch_w, var_w: float = 0.0, spec_w: float = 0.0, heavy=None, heavy_w: float = 1.0):
    """pred/true [B,K,6,H,W] anomalies. Latitude-weighted squared error per channel, plus (spec v1):
    RAIN-2 heavy-rain cells count heavy_w x on the rain channel; SHP-1 spread loss; SHP-2 spectral loss."""
    se = (pred - true) ** 2 * lat_w
    if heavy is not None and heavy_w != 1.0:
        se = se.clone()
        se[:, :, TP] = se[:, :, TP] * (1.0 + (heavy_w - 1.0) * heavy.to(se.dtype))
    loss = (se.mean(dim=(0, 1, 3, 4)) * ch_w).sum() / ch_w.sum()
    if var_w > 0:                                                      # SHP-1: keep spatial spread (anti-blur)
        sp, st = pred.std(dim=(-2, -1)), true.std(dim=(-2, -1))
        loss = loss + var_w * ((sp - st).abs().mean(dim=(0, 1)) * ch_w).sum() / ch_w.sum()
    if spec_w > 0:                                                     # SHP-2: match detail at every scale
        ap, at = torch.fft.rfft2(pred).abs(), torch.fft.rfft2(true).abs()
        d = (ap - at).abs().mean(dim=(0, 1, 3, 4)) / at.mean(dim=(0, 1, 3, 4)).clamp_min(1e-6)
        loss = loss + spec_w * (d * ch_w).sum() / ch_w.sum()
    return loss


class PhysScale:
    """Turns anomaly differences into physical differences in units of the judge's channel std (SHP-8)."""

    def __init__(self, codec: Codec, device):
        mean, std = load_stats()
        self.mean = torch.tensor(mean, device=device); self.std = torch.tensor(std, device=device)
        self.anom = torch.tensor(codec.anom_std, dtype=torch.float32, device=device)
        self.sqrt = codec.rain_mode == "sqrt"

    def rain_mm(self, z_tp, cf_tp):
        clim_mm = torch.relu(cf_tp * self.std[TP] + self.mean[TP])
        if self.sqrt:
            return torch.relu(z_tp * self.anom[TP] + torch.sqrt(clim_mm)) ** 2
        return torch.relu(z_tp * self.anom[TP] + clim_mm)

    def diff_std(self, za, zb, cf):
        """[B,K,6,H,W] anomalies (a, b) and target-day climatology features -> |a - b| in judge std units."""
        d = (za - zb).abs() * (self.anom / self.std)[None, None, :, None, None]
        rain = (self.rain_mm(za[:, :, TP], cf[:, :, TP]) - self.rain_mm(zb[:, :, TP], cf[:, :, TP])).abs() / self.std[TP]
        return torch.cat([d[:, :, :TP], rain[:, :, None], d[:, :, TP + 1:]], dim=2)


def shock_penalty(net, hist, cf, sc, fixed, unroll, rcfg, pred, ps: PhysScale):
    """SHP-8: warm-ocean-style shock (+3 K, +15% water vapour) in a random 12x12 box on the last input day;
    penalise any response above SHOCK_LIMIT judge-std (the judge fails a shock above 5)."""
    h, w = hist.shape[-2:]                                             # full grid or a V2-E6 crop
    i0 = np.random.randint(0, h - SHOCK_BOX + 1); j0 = np.random.randint(0, w - SHOCK_BOX + 1)
    box = torch.zeros(h, w, device=hist.device); box[i0:i0 + SHOCK_BOX, j0:j0 + SHOCK_BOX] = 1.0
    last = hist[:, -1]
    tcwv_phys = last[:, TCWV] * ps.anom[TCWV] + (cf[:, 0, TCWV] * ps.std[TCWV] + ps.mean[TCWV])
    pert = torch.zeros_like(last)
    pert[:, T2M] = box * (3.0 / ps.anom[T2M])
    pert[:, TCWV] = box * (0.15 * tcwv_phys / ps.anom[TCWV])
    hs = torch.cat([hist[:, :2], (last + pert)[:, None]], dim=1)
    shocked = rollout(net, hs, cf, sc, fixed, unroll, rcfg, mode="train")
    return (torch.relu(ps.diff_std(shocked, pred, cf) - SHOCK_LIMIT) ** 2).mean()


def parse_net_config(s: str) -> dict:
    """'{"width": 48}' (JSON) or 'width=48,rounds=4' (shell friendly, no quotes needed)."""
    s = s.strip()
    if not s or s.startswith("{"):
        return json.loads(s or "{}")
    out = {}
    for part in s.split(","):
        k, v = part.split("=", 1)
        out[k.strip()] = json.loads(v)          # numbers and true/false parse; bare words need quotes
    return out


def parse_curriculum(s: str) -> list[tuple[int, int]]:
    return [(int(a), int(b)) for a, b in (p.split(":") for p in s.split(","))]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--net", default="{}", help='network kwargs: JSON or k=v list, e.g. width=48,rounds=4')
    # defaults = spec v1 recipe (docs/SPEC.md); sweep 1 used: --rain none --var-loss 0 --spec-loss 0 --heavy-weight 1
    ap.add_argument("--rain", default="sqrt", choices=["none", "sqrt"])                  # REP-4
    ap.add_argument("--curriculum", default="1:4,2:3,4:3,8:2")                          # ROL-2
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--var-loss", type=float, default=0.1)                              # SHP-1
    ap.add_argument("--spec-loss", type=float, default=0.05)                            # SHP-2
    ap.add_argument("--heavy-weight", type=float, default=3.0)                          # RAIN-2
    # phase 2 options (spec 1.1)
    ap.add_argument("--edge-relax", action="store_true", help="EDG-2 F4: pull edge cells toward normal")
    ap.add_argument("--latlon", action="store_true", help="EDG-2 F6: lat/lon input maps (weight decay 1e-3)")
    ap.add_argument("--rain-head", default="none", choices=["none", "two_part"], help="RAIN-3 D4")
    ap.add_argument("--rain-bce", type=float, default=0.5, help="RAIN-3 weight of the rain/no-rain loss")
    ap.add_argument("--shock-loss", type=float, default=0.0, help="SHP-8 weight of the shock stability guard")
    ap.add_argument("--random-crop", action="store_true", help="V2-E6 (b): train on random 24x28..32x36 crops")
    ap.add_argument("--weight-decay", type=float, default=None, help="default 1e-4, or 1e-3 with --latlon")
    ap.add_argument("--channel-weights", default="1,1,1,1,1,1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--resume", action="store_true", help="continue from runs/ckpt/<name>_last.pt if it exists")
    a = ap.parse_args(argv)
    if a.rain_head == "two_part" and a.rain != "sqrt":
        raise SystemExit("--rain-head two_part needs --rain sqrt (spec RAIN-3)")

    torch.manual_seed(a.seed); np.random.seed(a.seed); torch.set_num_threads(a.threads)
    dev = torch.device(a.device)
    codec = Codec(rain_mode=a.rain)
    codec.anom_std = anomaly_std(a.rain)
    T = Tensors(codec, dev, a.latlon)
    mean, std = load_stats()
    rcfg = {}
    if a.edge_relax:
        rcfg["edge_relax"] = list(EDGE_RELAX_DEFAULT)
    if a.rain_head == "two_part":
        rcfg["rain_head"] = "two_part"
        rcfg["rain_ref"] = {"mean": float(mean[TP]), "std": float(std[TP]), "anom_std": float(codec.anom_std[TP])}
    net_cfg = parse_net_config(a.net)
    net = build_net(a.family, n_inputs(a.latlon), 7 if a.rain_head == "two_part" else 6, **net_cfg).to(dev)
    n_par = sum(p.numel() for p in net.parameters())
    wd = a.weight_decay if a.weight_decay is not None else (1e-3 if a.latlon else 1e-4)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=wd)
    cur = parse_curriculum(a.curriculum)
    total_steps = sum(n_ep * -(-len(window_starts(T.ds, TRAIN_YEARS, u)) // a.batch) for u, n_ep in cur)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total_steps + 1, pct_start=0.05)
    lat_w = torch.tensor(np.cos(np.deg2rad(LAT)) / np.cos(np.deg2rad(LAT)).mean(), dtype=torch.float32, device=dev)[:, None]
    ch_w = torch.tensor([float(x) for x in a.channel_weights.split(",")], device=dev)
    ps = PhysScale(codec, dev) if a.shock_loss > 0 else None
    bce = torch.nn.BCEWithLogitsLoss()
    print(f"{a.family} {net_cfg} params={n_par:,} device={dev} rain={a.rain} curriculum={cur} "
          f"options: edge_relax={a.edge_relax} latlon={a.latlon} rain_head={a.rain_head} shock_loss={a.shock_loss} wd={wd} random_crop={a.random_crop}")

    best, out_path = float("inf"), ARTIFACTS / "models" / f"{a.name}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    state_path = ROOT / "runs" / "ckpt" / f"{a.name}_last.pt"         # full training state, saved every epoch
    state_path.parent.mkdir(parents=True, exist_ok=True)
    step_count, ep, done_ep = 0, 0, 0
    if a.resume and state_path.exists():
        st = torch.load(state_path, map_location=dev, weights_only=False)
        if st.get("args") and (st["args"]["curriculum"], st["args"]["batch"]) != (a.curriculum, a.batch):
            raise SystemExit(f"--resume needs the same --curriculum/--batch as the original run: {st['args']['curriculum']}")
        net.load_state_dict(st["net"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        step_count, done_ep, best = st["step_count"], st["epoch"], st["best"]
        np.random.set_state(st["np_rng"]); torch.set_rng_state(st["torch_rng"])
        print(f"resumed {a.name} after epoch {done_ep} (best val {best:.4f})", flush=True)
    for unroll, n_ep in cur:
        tr_idx = window_starts(T.ds, TRAIN_YEARS, unroll)
        va_idx = window_starts(T.ds, VAL_YEARS, unroll, stride=3)
        for _ in range(n_ep):
            ep += 1
            if ep <= done_ep:                                            # finished before the restart
                continue
            t0 = time.time(); net.train(); tot = 0.0
            perm = np.random.permutation(tr_idx)
            for bi, i in enumerate(range(0, len(perm), a.batch)):
                hist, cf, sc, true, heavy, wet = T.batch(perm[i:i + a.batch], unroll)
                fixed, lw = T.fixed, lat_w
                if a.random_crop:                                        # V2-E6 (b); validation stays full grid
                    (hist, cf, true, heavy, wet), fixed, lw = random_crop([hist, cf, true, heavy, wet], T.fixed, lat_w)
                pred, logits = rollout(net, hist, cf, sc, fixed, unroll, rcfg, mode="train", return_logits=True)
                loss = loss_fn(pred, true, lw, ch_w, a.var_loss, a.spec_loss, heavy, a.heavy_weight)
                if logits is not None:                                   # RAIN-3: learn "will it rain?"
                    loss = loss + a.rain_bce * bce(logits, wet.to(logits.dtype))
                if ps is not None and bi % 4 == 0:                       # SHP-8: every 4th batch
                    loss = loss + a.shock_loss * shock_penalty(net, hist, cf, sc, fixed, unroll, rcfg, pred, ps)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                if step_count < total_steps:
                    sched.step(); step_count += 1
                tot += loss.item() * len(hist)
            net.eval(); vl = 0.0
            with torch.no_grad():
                for i in range(0, len(va_idx), 128):
                    hist, cf, sc, true, _, _ = T.batch(va_idx[i:i + 128], unroll)
                    pv = rollout(net, hist, cf, sc, T.fixed, unroll, rcfg, mode="infer")
                    vl += loss_fn(pv, true, lat_w, ch_w).item() * len(hist)          # plain MSE, inference mode
            vl /= len(va_idx)
            print(f"ep {ep:3d} unroll {unroll}  train {tot / len(tr_idx):.4f}  val {vl:.4f}  {time.time() - t0:.0f}s", flush=True)
            if unroll == cur[-1][0] and vl < best:
                best = vl
                torch.save({"family": a.family, "net_config": net_cfg, "codec": codec.to_dict(),
                            "inputs": {"latlon": a.latlon}, "rollout": rcfg,
                            "state_dict": {k: v.cpu() for k, v in net.state_dict().items()},
                            "train_args": vars(a)}, out_path)
            tmp = state_path.with_suffix(".tmp")                         # write then rename: a power cut never
            torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),   # leaves a broken file
                        "step_count": step_count, "epoch": ep, "best": best, "args": vars(a),
                        "np_rng": np.random.get_state(), "torch_rng": torch.get_rng_state()}, tmp)
            tmp.replace(state_path)
    print("saved", out_path, "best val", round(best, 4))
    if not a.no_eval:
        from .evaluate import evaluate
        res = evaluate({"model": "neural", "checkpoint": str(out_path.relative_to(ARTIFACTS))}, "val", a.name)
        print(json.dumps({k: res[k] for k in ("auto_events_of_70", "auto_random_of_70", "gate")}),
              "SS", round(res["random"]["mean_ss"], 3), "CSI", round(res["random"]["mean_csi_skill"], 3),
              "blur", res["random"]["blur_ratio"], "causal", round(res["shocks"]["causal_pts"], 2))


if __name__ == "__main__":
    main()
