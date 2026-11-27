"""V2-CYC-ALL extras: reach on firing windows, shock maxima, worst-case timing on 2 and 4 threads (side by side)."""
import os, sys, time, json, numpy as np
TH = int(sys.argv[1]); os.environ["TWIN_THREADS"] = str(TH)
sys.path.insert(0, r"C:/Users/vangs/cursor/devforge_2/wt-cyc")
from monsoon_r1 import contract as C, scoring as S
from training.data import load, VAL_YEARS, TEST_YEARS
from training.events import pick_events, named_windows
from training.evaluate import probe_spread
from twin.config import load_stats
from twin.engine import Engine
from twin.models.cyclone import detect
ds = load(); _, std = load_stats()
spec_all = json.load(open(r"C:/Users/vangs/cursor/devforge_2/wt-cyc/artifacts/cyclone_a_all.json"))
engs = {"final_cnn": Engine({"model": "neural", "checkpoint": "models/final_cnn.pt"}), "cyc_a_all": Engine(spec_all)}
wins = {"val cyclone": pick_events(ds, VAL_YEARS)["cyclone"], "test cyclone": pick_events(ds, TEST_YEARS)["cyclone"],
        "Fani": named_windows(ds, "val")["cyclone_fani"], "Amphan": named_windows(ds, "test")["cyclone_amphan"],
        "Tauktae": named_windows(ds, "test")["cyclone_tauktae"], "Yaas": named_windows(ds, "test")["cyclone_yaas"]}
if TH == 4:
    for lab, t in wins.items():
        c = detect(ds.data[t], float(ds.doy[t]), spec_all["threshold_pa"])
        r = {}
        for k, e in engs.items():
            r[k] = max(probe_spread(e, ds, [t], std, cell)["rmax_day1"] for cell in (c, (c[0], c[1] + 5), (16, 18)))
        init = ds.data[t-2:t+1]; sc = C.doy_sincos(int(ds.doy[t])); sh = {}
        for k, e in engs.items():
            b = e.simulate(init, sc, 10); z = e.simulate(init, sc, 10, np.zeros((6, 32, 36), "f4"))
            assert np.array_equal(b, z), "null test not exact"
            o = e.simulate(init, sc, 10, S.make_shock("warm_ocean", init[-1], std))
            sh[k] = round(float((np.abs(o - b) / std[None, :, None, None]).max()), 2)
        print(f"{lab:13s} centre {c}  reach rmax day1 (max of centre/edge/default) {r}  warm_ocean max std {sh}  null exact")
res = {k: [] for k in engs}
for rep in range(10):
    for lab, t in wins.items():
        init = ds.data[t-2:t+1]; sc = C.doy_sincos(int(ds.doy[t])); shock = S.make_shock("warm_ocean", init[-1], std)
        for k, e in engs.items():
            for p in (None, shock):
                t0 = time.perf_counter(); e.simulate(init, sc, 10, p); res[k].append((time.perf_counter() - t0) * 1000)
print(f"{TH} threads, 10-day rollouts on the 6 firing windows (null + warm_ocean, 10 reps, interleaved): "
      + ", ".join(f"{k} median {np.median(v):.0f} / p99 {np.percentile(v, 99):.0f} / max {np.max(v):.0f} ms" for k, v in res.items()))
