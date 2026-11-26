"""Fit post-processing for one base model (spec ROL-3, SHP-3, SHP-4, SHP-7) and choose what to switch on.

    python -m training.fit_post --checkpoint models/r2_cnn.pt --name r2_cnn
Writes artifacts/post/<name>.npz (see twin/models/post.py for the layout).

1. Fit on TRAINING-year forecasts (400 random windows):
   blend weights (least squares toward the damped forecast), rescale factors (spread of truth / spread of
   forecast, clipped to [1, 2]), rain quantile tables per season and lead day (101 levels).
2. Choose on VALIDATION-year forecasts (events + 60 random windows), scored with the judge's formulas
   (error skill + heavy rain + anti-blur; shocks are not affected by these steps and are scored later):
   rescale channel by channel (greedy, non-rain only), then the rain map on/off.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from monsoon_r1 import contract as C
from monsoon_r1 import scoring as S

from twin.climatology import clim_sequence
from twin.config import ARTIFACTS, TP
from twin.engine import Engine
from twin.models import build_model
from twin.models.post import season_of

from .data import TRAIN_YEARS, VAL_YEARS, load
from .events import pick_events, random_windows

STEPS = C.SCORE_STEPS
LEADS = [L - 1 for L in C.LEAD_DAYS]
Q = np.linspace(0.0, 1.0, 101)


def collect(engine: Engine, damped, ds, idx):
    """Run base and damped forecasts on windows -> dict of stacked arrays [W, steps, 6, H, W]."""
    out = {"pred": [], "damped": [], "truth": [], "clim": [], "init": [], "doy": []}
    for t in idx:
        init, doy = ds.data[t - 2:t + 1], float(ds.doy[t])
        out["pred"].append(engine.simulate(init, C.doy_sincos(int(doy)), STEPS))
        out["damped"].append(damped.rollout(init, doy, STEPS))
        out["truth"].append(ds.data[t + 1:t + 1 + STEPS]); out["clim"].append(clim_sequence(doy, STEPS))
        out["init"].append(init); out["doy"].append(doy)
    return {k: np.asarray(v) for k, v in out.items()}


def fit_blend(d) -> np.ndarray:
    m, dm, y = d["pred"], d["damped"], d["truth"]
    a, b = dm - m, y - m
    w = (a * b).sum(axis=(0, 3, 4)) / np.maximum((a * a).sum(axis=(0, 3, 4)), 1e-12)     # [steps, 6]
    return np.clip(w, 0.0, 1.0).astype("f4")


def blend(d, w):
    w = w[None, :, :, None, None]
    return (1 - w) * d["pred"] + w * d["damped"]


def fit_scale(pred, d) -> np.ndarray:
    sp = (pred - d["clim"]).std(axis=(3, 4)).mean(axis=0)
    st = (d["truth"] - d["clim"]).std(axis=(3, 4)).mean(axis=0)
    s = np.clip(st / np.maximum(sp, 1e-6), 1.0, 2.0).astype("f4")
    s[:, TP] = 1.0                                                   # rain is handled by the quantile map
    return s


def rescale(pred, d, s, on):
    f = np.where(on[None, :], s, 1.0)[None, :, :, None, None]
    return d["clim"] + (pred - d["clim"]) * f


def fit_qmap(pred, d):
    q_fc = np.zeros((4, STEPS, len(Q)), "f4"); q_tr = np.zeros_like(q_fc)
    seasons = np.array([[season_of(doy + k + 1) for k in range(STEPS)] for doy in d["doy"]])   # [W, steps]
    for s in range(4):
        for k in range(STEPS):
            sel = seasons[:, k] == s
            if not sel.any():
                q_fc[s, k] = Q; q_tr[s, k] = Q; continue
            q_fc[s, k] = np.quantile(np.maximum(pred[sel, k, TP], 0), Q)
            q_tr[s, k] = np.quantile(d["truth"][sel, k, TP], Q)
    q_fc = np.maximum.accumulate(q_fc, axis=-1) + np.arange(len(Q)) * 1e-6   # strictly increasing for np.interp
    return q_fc.astype("f4"), q_tr


def rainmap(pred, d, q_fc, q_tr):
    out = pred.copy()
    for w, doy in enumerate(d["doy"]):
        for k in range(STEPS):
            s = season_of(doy + k + 1)
            out[w, k, TP] = np.interp(np.maximum(out[w, k, TP], 0), q_fc[s, k], q_tr[s, k])
    return out


def points(pred, d) -> float:
    """Judge formulas without shocks: RMSE skill (30) + heavy-rain CSI (15) + anti-blur (10)."""
    ss, cs, ratios = [], [], []
    for w in range(len(pred)):
        p, y = pred[w], d["truth"][w]
        per = np.repeat(d["init"][w][-1][None], STEPS, axis=0); cl = d["clim"][w]
        ss.append(float(np.nanmean(S.skill_vs_best_ref(p[LEADS], y[LEADS], per[LEADS], cl[LEADS])["ss"])))
        cs.append(S.csi_skill(S.csi(p[LEADS, TP], y[LEADS, TP]), S.csi(per[LEADS, TP], y[LEADS, TP]),
                              S.csi(cl[LEADS, TP], y[LEADS, TP])))
        ratios.append(S.variance_ratio(p[7], y[7]))
    fp = S.forecast_points(ss, cs)
    return fp["points"] + S.anti_blur_points(np.array(ratios))


def main(argv=None):
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--checkpoint"); g.add_argument("--spec")
    ap.add_argument("--name", required=True)
    ap.add_argument("--n-train", type=int, default=400)
    a = ap.parse_args(argv)
    base = {"model": "neural", "checkpoint": a.checkpoint} if a.checkpoint else json.loads(open(a.spec).read())
    ds = load(); engine = Engine(base); damped = build_model({"model": "damped_anomaly"})

    tr = collect(engine, damped, ds, random_windows(ds, TRAIN_YEARS, n=a.n_train, seed=1))
    w = fit_blend(tr); pb = blend(tr, w)
    s = fit_scale(pb, tr)
    on_all = np.array([c != TP for c in range(6)])
    q_fc, q_tr = fit_qmap(rescale(pb, tr, s, on_all), tr)

    va = collect(engine, damped, ds, list(pick_events(ds, VAL_YEARS).values()) + random_windows(ds, VAL_YEARS))
    log = {"raw": points(va["pred"], va), "blend": points(blend(va, w), va)}
    blend_on = bool(log["blend"] > log["raw"] + 1e-6)                # ROL-3 (spec 1.1): keep only if it helps
    vb = blend(va, w) if blend_on else va["pred"]
    on = np.zeros(6, bool); best = max(log["raw"], log["blend"])
    for c in range(6):                                               # greedy per-channel rescale (SHP-3)
        if c == TP:
            continue
        trial = on.copy(); trial[c] = True
        p = points(rescale(vb, va, s, trial), va)
        if p > best + 1e-6:
            on, best = trial, p
    log["rescale"] = best
    vr = rescale(vb, va, s, on)
    p_map = points(rainmap(vr, va, q_fc, q_tr), va)
    rain_on = bool(p_map > best + 1e-6)                              # SHP-4: keep only if it helps
    log["rainmap"] = p_map

    out = ARTIFACTS / "post" / f"{a.name}.npz"; out.parent.mkdir(exist_ok=True)
    np.savez_compressed(out, blend_w=w, scale=s, scale_on=on, q_fc=q_fc, q_true=q_tr, rainmap_on=np.array(rain_on),
                        blend_on=np.array(blend_on))
    print(f"wrote {out}")
    print("blend weight lead1/8:", w[0].round(2), w[7].round(2))
    print("rescale lead8:", s[7].round(2), "on:", [n for n, o in zip(C.CHANNELS, on) if o])
    print("val points (no shocks, /55):", {k: round(v, 2) for k, v in log.items()}, "blend_on:", blend_on,
          "rainmap_on:", rain_on)


if __name__ == "__main__":
    main()
