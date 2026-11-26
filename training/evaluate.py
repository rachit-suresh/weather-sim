"""Local grader: scores any model spec exactly the way the judge does, using the kit's own scoring code
(monsoon_r1.scoring), on held-out years. Appends one row per run to runs/leaderboard.csv.

    python -m training.evaluate --model damped_anomaly
    python -m training.evaluate --checkpoint models/cnn_base.pt --split val
    python -m training.evaluate --spec artifacts/active.json --split test

Two window sets per split:
  events  5 windows like the hidden ones (noisy: only 5 samples, same as the judge)
  random  60 random windows (stable estimate of average skill)
Shock tests (warm_ocean, monsoon_stall, null) run on every event window and are averaged.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from monsoon_r1 import contract as C
from monsoon_r1 import scoring as S

from twin.climatology import clim_sequence
from twin.config import load_stats
from twin.engine import Engine

from .data import ROOT, TEST_YEARS, VAL_YEARS, load
from .events import pick_events, random_windows

LEADS = [L - 1 for L in C.LEAD_DAYS]       # rollout indices for lead days 1, 4, 8
STEPS = C.SCORE_STEPS


def _window(ds, t):
    return ds.data[t - 2:t + 1], ds.doy[t], ds.data[t + 1:t + 1 + STEPS]


def score_windows(engine: Engine, ds, idx: list[int]) -> dict:
    ss, ss_p4, csi_sk, ratios, ms = [], [], [], [], []
    for t in idx:
        init, doy, truth = _window(ds, t)
        t0 = time.perf_counter()
        pred = engine.simulate(init, C.doy_sincos(int(doy)), STEPS)
        ms.append((time.perf_counter() - t0) * 1000)
        if S.validity(pred, STEPS):
            ss.append(0.0); continue                               # judge: invalid window scores 0
        persist = np.repeat(init[-1][None], STEPS, axis=0)
        clim = clim_sequence(float(doy), STEPS)
        r = S.skill_vs_best_ref(pred[LEADS], truth[LEADS], persist[LEADS], clim[LEADS])
        ss.append(float(np.nanmean(r["ss"])))
        r4 = S.skill_vs_best_ref(pred[[3]], truth[[3]], persist[[3]], clim[[3]])
        ss_p4.append(float(np.nanmean(r4["ss_persist"])))
        c_m = S.csi(pred[LEADS, 1], truth[LEADS, 1]); c_p = S.csi(persist[LEADS, 1], truth[LEADS, 1])
        c_c = S.csi(clim[LEADS, 1], truth[LEADS, 1])
        csi_sk.append(S.csi_skill(c_m, c_p, c_c))
        ratios.append(S.variance_ratio(pred[7], truth[7]))
    fp = S.forecast_points(ss, csi_sk)
    ratios = np.array(ratios) if ratios else np.full((1, 6), np.nan)
    return {"n": len(idx), "mean_ss": fp["mean_ss"], "mean_csi_skill": fp["mean_csi_skill"],
            "rmse_pts": fp["rmse_pts"], "csi_pts": fp["csi_pts"], "blur_pts": S.anti_blur_points(ratios),
            "blur_ratio": np.nanmean(ratios, axis=0).round(3).tolist(),
            "ss_persist_day4": float(np.mean(ss_p4)) if ss_p4 else float("nan"), "ms_per_rollout": float(np.mean(ms))}


def score_shocks(engine: Engine, ds, idx: list[int], std: np.ndarray) -> dict:
    pts, checks = [], []
    for t in idx:
        init, doy, _ = _window(ds, t)
        sc = C.doy_sincos(int(doy))
        base = engine.simulate(init, sc, STEPS)
        res = []
        for name, box in (("warm_ocean", C.BOX_BAY_OF_BENGAL), ("monsoon_stall", C.BOX_ARABIAN_SEA)):
            out = engine.simulate(init, sc, STEPS, S.make_shock(name, init[-1], std))
            res.append(S.causality_checks(base, out, box, std))
        null = engine.simulate(init, sc, STEPS, S.make_shock("null", init[-1], std))
        null_ok = S.null_consistent(base, null, std)
        pts.append(S.causality_points(res, null_ok))
        checks.append({"warm_ocean": {k: res[0][k] for k in ("nonzero", "localized", "propagates", "stable")},
                       "monsoon_stall": {k: res[1][k] for k in ("nonzero", "localized", "propagates", "stable")},
                       "null_ok": null_ok})
    return {"causal_pts": float(np.mean(pts)), "checks": checks}


def probe_spread(engine: Engine, ds, idx: list[int], std: np.ndarray, cell=(16, 18)) -> dict:
    """Effective receptive field, measured: a 1-cell shock (+0.5 std on every channel) at `cell`.
    r95  = radius (cells) holding 95% of the response energy;  rmax = farthest cell that moved at all
    (|delta| > 1e-4 std). The judge's day-k 'teleport' radius is 3 + 8k cells from the shock box."""
    ii, jj = np.indices((C.NLAT, C.NLON))
    dist = np.sqrt((ii - cell[0]) ** 2 + (jj - cell[1]) ** 2)
    order = np.argsort(dist.ravel())
    out = {"r95_day1": [], "rmax_day1": [], "r95_day4": [], "rmax_day4": []}
    for t in idx:
        init, doy, _ = _window(ds, t)
        sc = C.doy_sincos(int(doy))
        pert = np.zeros((C.NC, C.NLAT, C.NLON), "f4"); pert[:, cell[0], cell[1]] = 0.5 * std
        d = (engine.simulate(init, sc, STEPS, pert) - engine.simulate(init, sc, STEPS)) / std[None, :, None, None]
        for k, lab in ((0, "day1"), (3, "day4")):
            e = (d[k] ** 2).sum(0).ravel()
            cum = np.cumsum(e[order]) / max(e.sum(), 1e-20)
            out[f"r95_{lab}"].append(float(dist.ravel()[order][np.searchsorted(cum, 0.95)]) if e.sum() > 0 else 0.0)
            moved = np.abs(d[k]).max(0) > 1e-4
            out[f"rmax_{lab}"].append(float(dist[moved].max()) if moved.any() else 0.0)
    return {k: round(float(np.mean(v)), 2) for k, v in out.items()}


def evaluate(spec: dict, split: str = "val", tag: str = "", write: bool = True) -> dict:
    ds = load()
    years = VAL_YEARS if split == "val" else TEST_YEARS
    engine = Engine(spec)
    _, std = load_stats()
    ev = pick_events(ds, years)
    rnd = random_windows(ds, years)
    e = score_windows(engine, ds, list(ev.values()))
    r = score_windows(engine, ds, rnd)
    s = score_shocks(engine, ds, list(ev.values()), std)
    spread = probe_spread(engine, ds, list(ev.values())[:3], std)
    auto_events = e["rmse_pts"] + e["csi_pts"] + e["blur_pts"] + s["causal_pts"]
    auto_random = r["rmse_pts"] + r["csi_pts"] + r["blur_pts"] + s["causal_pts"]
    gate = bool(e["ss_persist_day4"] > 0 and any(c["warm_ocean"]["localized"] or c["monsoon_stall"]["localized"]
                                                for c in s["checks"]))
    res = {"time": datetime.now().isoformat(timespec="seconds"), "tag": tag, "model": engine.name, "split": split,
           "auto_events_of_70": round(auto_events, 2), "auto_random_of_70": round(auto_random, 2), "gate": gate,
           "events": e, "random": r, "shocks": s, "spread": spread, "event_windows": {k: int(v) for k, v in ev.items()}}
    if write:
        _append(res)
    return res


def _append(res: dict):
    runs = ROOT / "runs"; runs.mkdir(exist_ok=True)
    (runs / "eval").mkdir(exist_ok=True)
    name = f"{res['time'].replace(':', '')}_{res['model'].replace(':', '_')}_{res['split']}"
    (runs / "eval" / f"{name}.json").write_text(json.dumps(res, indent=1, default=float))
    row = {"time": res["time"], "tag": res["tag"], "model": res["model"], "split": res["split"],
           "auto_events_of_70": res["auto_events_of_70"], "auto_random_of_70": res["auto_random_of_70"],
           "gate": res["gate"], "ss_events": round(res["events"]["mean_ss"], 4), "ss_random": round(res["random"]["mean_ss"], 4),
           "csi_skill_events": round(res["events"]["mean_csi_skill"], 4), "csi_skill_random": round(res["random"]["mean_csi_skill"], 4),
           "blur_pts_events": round(res["events"]["blur_pts"], 2), "blur_pts_random": round(res["random"]["blur_pts"], 2),
           "causal_pts": round(res["shocks"]["causal_pts"], 2), "ms_per_rollout": round(res["random"]["ms_per_rollout"], 1),
           **res["spread"]}
    f = runs / "leaderboard.csv"
    rows = list(csv.DictReader(f.open(newline=""))) if f.exists() else []
    rows.append(row)
    fields = list(dict.fromkeys(k for r in rows for k in r))          # union of columns, first-seen order
    with f.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval=""); w.writeheader(); w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--model", help="persistence | climatology | damped_anomaly")
    g.add_argument("--checkpoint", help="path relative to artifacts/, e.g. models/cnn_base.pt")
    g.add_argument("--spec", help="json spec file")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    if a.model:
        spec = {"model": a.model}
    elif a.checkpoint:
        spec = {"model": "neural", "checkpoint": a.checkpoint}
    else:
        spec = json.loads(Path(a.spec).read_text())
    res = evaluate(spec, a.split, a.tag)
    e, r, s = res["events"], res["random"], res["shocks"]
    print(f"{res['model']} [{a.split}]  auto(events)={res['auto_events_of_70']}/70  auto(random)={res['auto_random_of_70']}/70  gate={res['gate']}")
    for lab, x in (("events", e), ("random", r)):
        print(f"  {lab:6s} SS={x['mean_ss']:.3f} ({x['rmse_pts']:.1f}/30)  CSI-skill={x['mean_csi_skill']:.3f} ({x['csi_pts']:.1f}/15)"
              f"  blur={x['blur_pts']:.1f}/10 ratios={x['blur_ratio']}  {x['ms_per_rollout']:.0f} ms")
    print(f"  shocks causality={s['causal_pts']:.2f}/15  windows={res['event_windows']}")
    print(f"  spread of a 1-cell shock (cells): {res['spread']}")


if __name__ == "__main__":
    main()
