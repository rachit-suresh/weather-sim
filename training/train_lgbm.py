"""Train family 2 (LightGBM per cell): one booster per channel predicts the next-day anomaly change.

    python -m training.train_lgbm --name lgbm_base
Writes artifacts/lgbm/<name>/{meta.json, model_<channel>.txt} and scores it with training.evaluate.
Trained on 1-step targets only (trees cannot be unrolled with gradients), so expect some drift.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from twin.calendar import clim_index, sincos
from twin.climatology import load_static
from twin.codec import Codec, clim_feature
from twin.config import ARTIFACTS, CHANNELS
from twin.lgbm_features import features
from twin.nets.common import edge_distance  # LightGBM keeps its own 3-map layout

from .data import TRAIN_YEARS, VAL_YEARS, load, window_starts
from .train_nn import anomaly_std


def build(ds, Z, table, idx, fixed):
    X = features(np.stack([Z[idx - 2], Z[idx - 1], Z[idx]], 1),
                 table[[clim_index(d) for d in ds.doy[idx + 1]]],
                 np.stack([sincos(d) for d in ds.doy[idx + 1].astype(float)]), fixed)
    Y = (Z[idx + 1] - Z[idx]).transpose(0, 2, 3, 1).reshape(-1, 6)
    return X, Y


def main(argv=None):
    import lightgbm as lgb
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="lgbm_base")
    ap.add_argument("--rain", default="none", choices=["none", "sqrt"])
    ap.add_argument("--stride", type=int, default=3, help="use every n-th training day (memory)")
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--leaves", type=int, default=63)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--no-eval", action="store_true")
    a = ap.parse_args(argv)

    ds = load()
    codec = Codec(rain_mode=a.rain, anom_std=anomaly_std(a.rain))
    doys = ds.doy.astype(float)
    Z = np.concatenate([codec.encode(ds.data[i:i + 512], doys[i:i + 512]) for i in range(0, len(doys), 512)])
    table = clim_feature(list(range(366)))
    st = load_static()
    fixed = np.concatenate([np.stack([st[0], st[1] / 1000.0]), edge_distance()]).astype("f4")
    Xt, Yt = build(ds, Z, table, window_starts(ds, TRAIN_YEARS, 1, a.stride), fixed)
    Xv, Yv = build(ds, Z, table, window_starts(ds, VAL_YEARS, 1, 7), fixed)
    print("train rows", Xt.shape, "val rows", Xv.shape)

    out = ARTIFACTS / "lgbm" / a.name
    out.mkdir(parents=True, exist_ok=True)
    params = {"objective": "regression", "learning_rate": a.lr, "num_leaves": a.leaves, "min_data_in_leaf": 200,
              "feature_fraction": 0.8, "bagging_fraction": 0.7, "bagging_freq": 1, "num_threads": a.threads,
              "verbose": -1, "seed": 0, "deterministic": True}
    for c, name in enumerate(CHANNELS):
        dt = lgb.Dataset(Xt, Yt[:, c], free_raw_data=False); dv = lgb.Dataset(Xv, Yv[:, c], reference=dt)
        b = lgb.train(params, dt, a.rounds, valid_sets=[dv], callbacks=[lgb.early_stopping(30, verbose=False)])
        b.save_model(str(out / f"model_{name}.txt"), num_iteration=b.best_iteration)
        print(f"{name}: best iter {b.best_iteration}  val l2 {b.best_score['valid_0']['l2']:.4f}", flush=True)
    (out / "meta.json").write_text(json.dumps({"codec": codec.to_dict(), "args": vars(a)}, indent=1))
    if not a.no_eval:
        from .evaluate import evaluate
        res = evaluate({"model": "lgbm", "dir": f"lgbm/{a.name}"}, "val", a.name)
        print(res["auto_events_of_70"], res["auto_random_of_70"])


if __name__ == "__main__":
    main()
