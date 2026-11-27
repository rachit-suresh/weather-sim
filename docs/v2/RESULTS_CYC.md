# v2 cyclone mode: r3_cnn everywhere, v2_e7c in a moving circle around a cyclone

Spec: `V2-CYC` (section 14, written before the code; revision 2 of its tracker rule is in the change log). Branch
`v2/cyclone-mode`, cut from `v2/e7-advect-correct`. No training: two existing checkpoints, `r3_cnn` (base) and
`v2_e7c` (mover).

## What was built

- `twin/models/cyclone.py`: model kind `cyclone_mode`, configured as
  `{"model": "cyclone_mode", "base": "models/r3_cnn.pt", "mover": "models/v2_e7c.pt", "variant": "A" | "B"}`.
  The spec files used for scoring are `runs/cyc_a.json` and `runs/cyc_b.json`. `artifacts/active.json` is unchanged
  (still `r3_cnn`).
- `TwinModel.rollout_shocked(init, base_states, doy, steps)`: the engine passes the unperturbed request states
  next to the shocked ones. The default ignores them, so every other model behaves exactly as before.
- **Detector** (unperturbed last input day only): msl anomaly vs the shipped climatology, over sea cells
  (land_sea_mask < 0.5) at lat <= 24N, in lon 80-97E (Bay of Bengal) or lon <= 77E (Arabian Sea). It fires when the
  basin minimum is below -THR.
- **Track:** `v2_e7c` is run with zero perturbation, and its msl-anomaly minimum is followed within 7 cells per day.
  The track is lost once the low is no deeper than LOST.
- **Mask:** the swath of the track so far. Each centre gets a circle with weight 1 out to R = 4 cells, then a cosine
  taper to 0 at 6 cells.
- **Output:** `mask * e7c + (1 - mask) * r3` on {u10, v10, msl} (variant **A**) or on all 6 channels (variant **B**).
  The output is r3 bit for bit when the detector does not fire.
- `training/cyclone_calib.py` (training-only code) computes THR, LOST, the firing rates, the composite radius
  profile and the tracker check. Its log is `runs/logs/v2_cyc_calib.log`.
- `tests/test_cyclone.py` checks:
  - mask shape;
  - no detection gives r3 bit for bit, both directly and through the engine, with and without a shock;
  - a shock that itself digs a deep low does NOT trigger the detector;
  - on a synthetic deep low: zero shock == no shock exactly; outside the mask the output is r3's exactly; under
    variant A, t2m / tp / tcwv are r3's everywhere.

## Calibration (training years 2005-2017 only)

- **THR** = the 98th percentile of the training-day basin minimum of -(msl anomaly) = **917.6 Pa (9.18 hPa)**.
  The deepest training cases are the expected ones: Oct 2013 (-21 hPa), Oct 2014 (-19), Dec 2011, May 2009,
  Apr 2006, Nov 2007, and so on.
- **LOST** = the training median of the same quantity = **192.1 Pa** (an ordinary day's deepest basin low).
- **R** comes from the composite of the 95 detected training days:

  | | centre | 1 cell | 2 | 3 | 4 | 5 | 6 | 8 |
  |---|---|---|---|---|---|---|---|---|
  | msl anomaly / centre value | 1.00 | 0.85 | 0.65 | 0.52 | 0.41 | 0.33 | 0.28 | 0.21 |
  | wind-speed anomaly (m/s) | +0.5 | +5.1 | +6.3 | +5.5 | +4.3 | +3.5 | +2.9 | +2.0 |

  The wind peaks at 2 cells and halves by 5-6 cells, so the mask has full weight out to 4 cells and reaches zero at
  6 cells.
- **Tracker check on the 43 training episodes.** Each cell gives the mean mask weight on days 1-3 at e7c's low, then
  at the true low, then the mask size in cells:

  | search \ lost | THR/2 (spec rev. 1) | LOST (training median) |
  |---|---|---|
  | 4 cells | 0.79 / 0.79, 145 cells (rev. 1) | 0.81 / 0.82, 149 |
  | 6 cells | | 0.83 / 0.85, 156 |
  | 7 cells | 0.84 / 0.86, 154 | **0.84 / 0.89, 159 (rev. 2)** |

**How often it fires:**

| | days | fires | share |
|---|---|---|---|
| train 2005-2017 | 4748 | 95 | 2.0% (by construction) |
| val 2018-19 | 730 | 21 | 2.9% |
| test 2020-21 | 731 | 18 | 2.5% |

**Judge windows that trigger the detector** (`evaluate` sets; the basin minimum is taken on the last input day):

| set | fired | not fired |
|---|---|---|
| val events | `cyclone` (last input 26 Oct 2019, -14.3 hPa at 16.5N 69.5E, Arabian Sea) | monsoon_onset, heavy_rain (-7.7), heatwave, winter_disturbance |
| val random | 1 of 60 | 59 |
| val named | `cyclone_fani` (-15.9 hPa) | kerala_floods, titli (-7.8), gaja (-4.7), vayu (-7.0), onset, heat |
| test events | `cyclone` (17 May 2020, -18.6 hPa) | the other 4 |
| test random | 0 of 60 | 60 |
| test named | `cyclone_amphan` (-25.4), `cyclone_tauktae` (-13.7), `cyclone_yaas` (-14.8) | `cyclone_nisarga` (-1.0) |

None of the shock-scored windows except the two `cyclone` event windows fires. The reach probe's windows (the first 3
event windows) and the 12 wind-shock windows never fire.

## Results

Rows for r3_cnn and v2_e7c are the `V2-P` numbers already in the leaderboard and in `RESULTS_E7.md`. cyc_a and cyc_b
were scored here with TWIN_THREADS=4, one evaluation at a time. Their leaderboard tags are `v2_cyc_a`,
`test_v2_cyc_a`, `v2_cyc_b` and `test_v2_cyc_b`, and their `runs/v2_checks.csv` rows are `v2_cyc_a` and `v2_cyc_b`.

| | r3_cnn | v2_e7c | cyc_a | cyc_b |
|---|---|---|---|---|
| val auto events / random / named | 67.51 / 68.12 / 68.12 | 67.15 / 67.38 / 67.38 | 67.51 / 68.12 / 68.12 | 67.49 / 68.12 / 68.12 |
| **val mean** | **67.92** | 67.30 | **67.92 (0.00)** | **67.91 (-0.01)** |
| test auto events / random / named | 68.12 / 67.65 / 68.12 | 65.42 / 66.03 / 67.38 | 68.12 / 67.65 / 68.12 | 68.12 / 67.65 / 68.12 |
| **test mean** | **67.96** | 66.28 | **67.96 (0.00)** | **67.96 (0.00)** |
| shocks val / test | 13.12 / 13.12 | 12.38 / 12.38 | 13.12 / 13.12 | 13.12 / 13.12 |
| warm_ocean stable val / test (of 5) | 5 / 4 | 3 / 3 | 5 / 4 | 5 / 4 |
| monsoon_stall propagates val / test | 0 / 1 | 2 / 1 | 0 / 1 | 0 / 1 |
| uncapped ss val events / random / named | .1681 / .1379 / .1854 | .1637 / .1342 / .1748 | .1681 / .1378 / **.1868** | .1674 / .1377 / **.1897** |
| uncapped ss test events / random / named | .1667 / .1181 / .1943 | .1569 / .1146 / .1921 | .1665 / .1181 / **.1970** | **.1673** / .1181 / **.1979** |
| csi_skill val events / random / named | .0976 / .1049 / .1404 | .1121 / .1086 / .1442 | .0976 / .1049 / .1404 | .0975 / .1051 / **.1422** |
| csi_skill test events / random / named | .1080 / .1053 / .1622 | .0870 / .1055 / .1554 | .1080 / .1053 / .1622 | **.1098** / .1053 / **.1677** |
| **Fani error day 1 / 2 / 3 (km)** | 350 / 245 / 809 | 350 / 0 / 607 | **350 / 0 / 607** | **350 / 0 / 607** |
| Fani mean days 1-3 (km) | 468 | 319 | **319 (-149)** | **319 (-149)** |
| **Fani landfall-latitude error (deg)** | +2.0 | 0.0 | **0.0** | **0.0** |
| Amphan day-2 error / landfall lat, **report only** | 786 km / -7.0 | 104 km / 0.0 | 104 km / 0.0 | 104 km / 0.0 |
| wind-shock west-coast rain (share < 0) | -1.29 mm (92%) | -1.98 mm (92%) | -1.29 mm (92%) | -1.29 mm (92%) |
| edge / interior day-4 RMSE (std) | .441 / .503 | .452 / .512 | .4415 / .5033 | .4418 / .5037 |
| reach rmax day 1 val / test (evaluate probe) | 9.90 / 9.67 | 8.72 / 8.81 | 9.90 / 9.67 | 9.90 / 9.67 |
| reach rmax day 1 on the 4 firing windows (probe at the centre, the mask edge and the default cell) | 9.22-9.90 | 8.25-9.22 | <= 9.90 | <= 9.90 |
| ms / 10-day rollout, 2 threads, side by side, no detection (median) | 93 | 158 | 95 | 98 |
| ms / 10-day rollout, 2 threads, side by side, detector fires (Fani; median / max) | 106 / 171 | 165 / 292 | 385 / 548 | 376 / 732 |

All timings were taken interleaved in one process: 15 repetitions of an unshocked and a warm_ocean-shocked call per
model. The machine was loaded (r3_cnn measured 37 ms in E7c). When the detector fires, cyclone mode runs three
networks: r3, e7c with zero shock (for the mask) and e7c shocked. An unshocked request reuses the mask run, so it
needs only two.

**Rain under the warm_ocean shock on the firing windows** (max |shocked - base| in judge std; tp is always the max):

| window | r3_cnn | v2_e7c | cyc_a | cyc_b |
|---|---|---|---|---|
| val cyclone event (Arabian Sea) | 2.04 | 2.74 | 2.04 | 2.04 |
| test cyclone event | 8.24 | 5.01 | 8.24 | 5.01 |
| Fani (named, not shock-scored) | 6.42 | **8.26** | 6.42 | **8.26** |
| Amphan (named, test) | 7.50 | **8.64** | 7.50 | **8.64** |

Variant A's rain response is r3's by construction. Variant B takes e7c's rain inside the mask, and with it e7c's
overshoot (Fani 8.26 vs 6.42). On the two scored cyclone event windows this happens not to cost a check: the val
window's storm is in the Arabian Sea, away from the Bay box, and on the test window r3 is already unstable.

## Verdict under V2-WIN (spec V2-CYC g)

| condition | cyc_a | cyc_b |
|---|---|---|
| Target: Fani mean days 1-3 at least 111 km lower (468 -> 319) and \|landfall lat\| not larger (2.0 -> 0.0) | pass | pass |
| val mean not lower by > 0.2 | pass (0.00) | pass (-0.01) |
| test mean not lower by > 0.2 | pass (0.00) | pass (0.00) |
| shocks not lower (13.12 / 13.12) | pass | pass |
| null test exact (null_ok 5/5 on every set; `test_cyclone` zero == none bit for bit) | pass | pass |
| reach rmax day 1 <= 11 | pass (9.90) | pass (9.90) |
| 10-day rollout < 1 s on 2 threads | pass (max 548 ms) | pass (max 732 ms) |
| repo scan 15/15 | pass | pass |
| `tests/test_engine.py` with the cyclone-mode spec | pass | pass |
| **verdict** | **WINS** | **WINS** |

**Both variants win, and variant A is the one to prefer.** It matches r3_cnn on every judge total and on every
uncapped column except named ss, which goes up. It moves the cyclone exactly as e7c does (Fani landfall latitude
exact, mean track error -149 km; Amphan, report only, lands exactly). It also cannot inherit e7c's warm_ocean rain
overshoot. Variant B has slightly better named-window skill (test csi .1677 vs .1622, ss .1979 vs .1970), but it does
inherit that overshoot (Fani 8.26 std). The only reason it costs no shock points is that no shock-scored window has a
Bay of Bengal cyclone with a stable r3 response.

**What this does and does not buy.** The judge's RMSE and CSI points are capped, and only 2 of 65 windows per split
fire, so the judge totals do not move at all. The gain is physical: on a cyclone window the forecast low now moves
and makes landfall at the right latitude instead of staying put. The detector also misses weaker named storms whose
1-degree daily low is shallower than 9.2 hPa on the last input day: Titli -7.8, Vayu -7.0, Gaja -4.7, Nisarga -1.0.
For those windows the output is r3_cnn.

`V2-WIN` gives winners 3 seeds; that does not apply here (no training; both parents are fixed checkpoints and the
combination is deterministic). Serving it would mean changing `artifacts/active.json` to the variant-A spec. That
was not done (not asked).

## Errors found

1. **Spec error (fixed before scoring, `V2-CYC` c, revision 2).** The first tracker rule (4-cell search, lost once
   the low is weaker than THR/2) assumed that `v2_e7c` carries the low within its 2.5 cells/day transport cap. On
   Fani (val, diagnostic only) e7c fills the low at 14.5N and rebuilds a weaker (-5 hPa) broad low 6.7 cells north
   on day 2. The track was lost on day 2, so cyclone mode output r3's track (350/245/809 km). The new rule is a
   7-cell search (a physical upper bound on daily translation) and LOST = the training median basin minimum. It was
   chosen on the 43 training episodes (table above), not on val or test scores. With it, cyclone mode reproduces
   e7c's Fani track.
2. **Measurement gap in the guard (spec, `V2-WIN` reach guard and `evaluate.probe_spread`).** The reach probe uses
   the first 3 event windows (monsoon_onset, heavy_rain, heatwave), and none of them fires the detector. For a
   conditional model, the evaluate reach column therefore says nothing about the blended path. It was measured
   separately on all 4 firing windows (<= 9.90). The spec should require the reach probe on windows where every
   path of the model is active.
3. **Spec gap (`V2-WIN`):** "only winners get extra seeds (3 total)" assumes a trained model. It does not apply to a
   training-free combination of fixed checkpoints. The rule should say so.
4. **Spec note (data rule, `V2-P` / `CON-6`):** the detector's anomaly uses the shipped climatology, which is built
   from 2005-2021 and so includes the val and test years. The threshold itself is set on training days only, as
   required. The same climatology is already used by the served model, the event picker and the judge-style
   anomalies, so this is not new to V2-CYC, but strictly the anomaly baseline is not training-only.
5. **Code (latent, known):** `runs/leaderboard.csv` has no lock. All 4 evaluations and 2 v2_checks runs were run one
   at a time (`runs/cyc_score.sh`); all rows are present.
6. **Risk, not an error:** on a firing window cyclone mode costs three network runs. Its worst rollout time was 732 ms
   on a loaded machine (2 threads), which leaves less margin under 1 s than r3_cnn (171 ms). The mask run could be
   cached per unperturbed request (the null request and the shocked request share it), but that adds state between
   calls, which `TwinModel` forbids, so it was not done.

No code errors (code not matching the spec) were found in the final version. `tests/test_cyclone.py` and
`tests/test_engine.py` pass with both variants and with the default spec, and `scripts/check_repo.py` gives 15/15.

---

# V2-CYC-ALL: all-data cyclone mode A candidate (kept ready, not served)

Spec: `V2-CYC-ALL` (written before the code). This is the `VAL-4` retrain of variant A. It is kept on
`v2/cyclone-mode` and has **not** been merged or served: that needs the user's personal approval.

- **Base:** `models/final_cnn.pt` (the v1.1 served model), copied read-only from `origin/main`. It is the same blob
  as on main (`bbb468a`).
- **Mover:** `models/v2_e7c_all.pt`. Warm start from `v2_e7c` (all 31 tensors), with the same E7c network.
  - Command: `train_nn --family advcnn --name v2_e7c_all --net '{"max_disp": 2.5, "interp": "bilinear",
    "block_k": [3, 3, 1]}' --rain-head two_part --curriculum 4:2,8:2 --lr 3e-4 --no-eval --resume
    --warm-start models/v2_e7c.pt --train-years 2005-2021 --final`.
  - Epochs 1-3 ran on 2 threads (930 / 820 / 1361 s). The machine then slept, and epoch 4 was resumed from the
    saved epoch-3 state on 4 threads (667 s).
  - The last-epoch weights are kept (`--final`). Log: `runs/logs/v2_e7c_all.log` plus `v2_e7c_all_resume.log`.
- **Detector:** the same rules, recomputed on all 2005-2021 days (`cyclone_calib.py --years 2005-2021`, log
  `runs/logs/v2_cyc_all_calib.log`): THR = 933.3 Pa (9.33 hPa) and LOST = 181.8 Pa. It fires on 1.9 / 2.5 / 2.3% of
  2005-17 / 2018-19 / 2020-21 days. The same judge windows fire as before: both `cyclone` event windows, Fani,
  Amphan, Tauktae, Yaas and 1 val random window.
- **Spec file:** `artifacts/cyclone_a_all.json`.

## Acceptance (as v1.1 `VAL-4`)

The scores are optimistic: 2018-21 is inside both models' training data. Both models were scored here side by side
with the same code, at 4 threads, one at a time. The final_cnn rows match main's v1.1 rows exactly.

| check | final_cnn | cyclone_a_all | rule | result |
|---|---|---|---|---|
| 2018-19 auto events / random / named (mean) | 67.39 / 67.75 / 67.75 (67.63) | 67.39 / 67.75 / 67.75 (67.63) | >= final - 0.2 | pass (0.00) |
| 2020-21 auto events / random / named (mean) | 68.88 / 68.88 / 68.88 (68.88) | 68.88 / 68.88 / 68.88 (68.88) | >= final - 0.2 | pass (0.00) |
| shocks 2018-19 / 2020-21 | 12.75 / 13.88 | 12.75 / 13.88 | not lower | pass |
| uncapped ss 2018-19 events / random / named | .2026 / .1631 / .2250 | .2035 / .1628 / .2258 | report | ~ |
| uncapped ss 2020-21 events / random / named | .2144 / .1521 / .2992 | .2119 / .1521 / .2872 | report | lower on test cyclones |
| csi_skill (all six columns) | .1252 / .1233 / .1835; .1295 / .1284 / .2310 | identical | report | = |
| Fani error d1 / d2 / d3 (km); landfall-lat error | 247 / 312 / 322; 0.0 | **247 / 104 / 232; 0.0** | lat not worse | pass |
| Amphan d2 error / landfall lat (report only) | 153 km / -1.0 | **104 km / 0.0** | report | better |
| wind-shock west-coast rain (share < 0) | -1.17 mm (83%) | -1.17 mm (83%) | report | = |
| null test exact (evaluate null_ok, and bit-for-bit zero == none on the 6 firing windows) | yes | yes | exact | pass |
| reach rmax day 1: evaluate probe / 6 firing windows (centre, mask edge, default cell) | 9.9 / 9.9 | 9.9 / 9.9 | <= 11 | pass |
| warm_ocean max response on the 6 firing windows (std) | 2.77 / 9.78 / 4.12 / 4.87 / 2.17 / 4.18 | identical (variant A keeps final_cnn's rain) | report | = |
| worst rollout, 6 firing windows, null + shocked, 10 reps, 4 threads (median / p99 / max) | 26 / 39 / 43 ms | 115 / 157 / **164 ms** | < 1 s | pass |
| same, 2 threads | 37 / 48 / 51 ms | 129 / 179 / **185 ms** | < 1 s | pass |
| `tests/test_engine.py` and `tests/test_cyclone.py` with `TWIN_MODEL_SPEC=artifacts/cyclone_a_all.json` (plus the defaults) | | ok | pass | pass |
| repo scan | | 15/15 | 15/15 | pass |
| fresh clone, plain git (no LFS): spec loads and simulates | | see below | loads | pass |

Logs: `runs/logs/v2_cyc_a_all_*.log`, `runs/logs/final_cnn_*.log`, `runs/logs/v2_cyc_a_all_extra.log` (script
`runs/cyc_all_extra.py`). Leaderboard tags: `final_cnn_val_cycbranch`, `final_cnn_test_cycbranch`,
`v2_cyc_a_all_val` and `v2_cyc_a_all_test`. v2_checks rows: `final_cnn` and `v2_cyc_a_all`.

**Reading.** The judge totals are unchanged, as with V2-CYC. The cyclone track gains survive the all-data retrain:
on Fani the day-2/3 errors fall from 312/322 km to 104/232 km, and on Amphan the landfall latitude becomes exact.
final_cnn has seen these storms in training, so its own tracks are already much better than r3_cnn's; even so,
cyclone mode improves them. One cost: uncapped ss on the 2020-21 named cyclone windows falls from .2992 to .2872.
The moved low (and its winds) is penalised on some of those windows, and the RMSE points stay capped at 30/30.

## How to serve it (only with the user's approval)

On the serving branch, replace the one line of `artifacts/active.json`. `models/v2_e7c_all.pt` and
`artifacts/cyclone_a_all.json` must be on that branch too, plus this branch's `twin/models/cyclone.py`,
`twin/models/__init__.py`, `twin/models/base.py` and `twin/engine.py`.

```
{"model": "cyclone_mode", "base": "models/final_cnn.pt", "mover": "models/v2_e7c_all.pt", "variant": "A", "threshold_pa": 933.3, "lost_pa": 181.8}
```

(or set the env `TWIN_MODEL_SPEC=artifacts/cyclone_a_all.json`, which needs no file edit). To check:
`TWIN_MODEL_SPEC=... python tests/test_engine.py`, `python tests/test_cyclone.py` and `python scripts/check_repo.py`;
`/health` then reports `cyclone_mode_A:final_cnn+v2_e7c_all`.

**To revert:** put back the v1.1 line

```
{"model": "neural", "checkpoint": "models/final_cnn.pt"}
```

(or unset `TWIN_MODEL_SPEC`) and restart the server. The engine change is backward compatible: every other model
ignores the unshocked states, so reverting the json alone is enough.

## Errors found in V2-CYC-ALL

1. **Code gap (fixed):** `v2/e7-advect-correct` was cut before main's `VAL-4` flags, so its `train_nn` had no
   `--train-years` / `--final`. They were ported unchanged from main; the only remaining difference from main's
   file is `--warm-start`.
2. **Process note:** the codec's anomaly std is recomputed on 2005-2021 (as `VAL-4` does), so the warm start begins
   under a slightly different normalisation than `v2_e7c` was trained with. The 4 fine-tune epochs absorb it. Not
   an error, but it means `v2_e7c_all` is not exactly "v2_e7c plus more data".
3. **Process (known hazard, `RESULTS_E7` error 1):** the machine slept during epoch 4. `--resume` restarted cleanly
   from the saved epoch-3 state, because that state had been written before the sleep.
