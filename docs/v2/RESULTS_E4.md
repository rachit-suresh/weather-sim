# V2-E4: AMSE loss (adjusted MSE on a 2-D DCT)

Date: 2026-11-27. Branch `v2/e4-amse`. Spec: `docs/SPEC.md` section 14 (`V2-P`, `V2-WIN`, `V2-E4` as refined on this
branch). Model: `artifacts/models/v2_e4.pt`. Training log: `runs/logs/v2_e4.log` (not committed).

## What changed (one thing)

`--loss amse` (default `--loss mse` = unchanged baseline). Training loss per sample, step and channel: orthonormal 2-D
DCT-II (32x32 and 36x36 matrices, two matmuls) of the anomaly map times sqrt(cos-lat weight); coefficients grouped into
50 radial bands b = round(sqrt((36 i / 32)^2 + j^2)); per band
`(sqrt(P) - sqrt(T))^2 + 2 max(P, T) (1 - X / sqrt(P T))`, summed over bands / (32 x 36). This replaces the
latitude-weighted MSE, the `SHP-1` spread loss and the `SHP-2` FFT-amplitude loss on all 6 channels (rain in sqrt
space). `RAIN-2` kept as a separate pointwise term (+2 x lat-weighted squared error at heavy-rain cells, so heavy
cells still count 3x). `RAIN-3` BCE, curriculum 1:4,2:3,4:3,8:2, train years 2005-2017, seed 0, checkpoint choice
(plain MSE on validation windows, last stage) all unchanged. Inference unchanged.

Unit checks (`tests/test_amse.py`): DCT matrices orthonormal; every coefficient in exactly one band; AMSE = 0 for
identical maps (1.5e-10); AMSE >= MSE and same scale (random error: MSE 0.249, AMSE 0.279); a pure amplitude error
gives AMSE = MSE; an all-zero (fully blurred) forecast costs 3x the MSE (3.01 vs 1.00), the anti-blur property;
heavy-rain term adds exactly 2x the heavy-cell error; gradients finite. Cost per epoch about the same as the baseline
loss (smoke: 15 s vs 20 s per epoch).

Run: epochs 1-8 on 1 thread, then (coordinator instruction, after the v1.1 retrain `final_cnn` ended) stopped
after epoch 8's state was written and resumed with `--threads 2` for epochs 9-12. Best = epoch 11
(val MSE 0.8669; epoch 12 0.8687).

## Scores (judge points /70; baseline `r3_cnn` rescored here and matches `V2-P` exactly)

| Set | r3_cnn events / random / named | **v2_e4** events / random / named | r3_cnn mean | **v2_e4 mean** | diff |
|---|---|---|---|---|---|
| val 2018-19 | 67.51 / 68.12 / 68.12 | 66.22 / 60.18 / 67.75 | 67.92 | **64.72** | **-3.20** |
| test 2020-21 | 68.12 / 67.65 / 68.12 | 67.38 / 54.28 / 67.38 | 67.96 | **63.01** | **-4.95** |

Point breakdown v2_e4 (RMSE /30, heavy rain /15, blur /10): val events 30.0 / 13.5 / 9.9, random 23.3 / 14.1 / 10.0,
named 30.0 / 15.0 / 10.0; test events 30.0 / 15.0 / 10.0, random **18.4** / 13.5 / 10.0, named 30.0 / 15.0 / 10.0.
The loss is almost all RMSE points on the 60 random windows.

## Uncapped numbers

| | r3_cnn val | v2_e4 val | r3_cnn test | v2_e4 test |
|---|---|---|---|---|
| ss events | 0.168 | 0.131 | 0.167 | 0.125 |
| ss random | 0.138 | **0.093** | 0.118 | **0.074** |
| ss named | 0.185 | 0.165 | 0.194 | 0.170 |
| csi_skill events | 0.098 | 0.090 | 0.108 | 0.116 |
| csi_skill random | 0.105 | 0.094 | 0.105 | 0.090 |
| csi_skill named | 0.140 | 0.135 | 0.162 | 0.173 |
| shocks /15 | 13.12 | **12.75** | 13.12 | **12.38** |
| reach r95 / rmax day 1 (cells) | 2.08 / 9.9 | 3.06 / 9.67 | 1.88 / 9.67 | 2.53 / 9.9 |
| ms per 10-day rollout (1 thread) | 133 | 136 | 137 | 133 |

## Day-8 variance ratios (pred / true), channels t2m / tp / u10 / v10 / msl / tcwv

From `runs/eval/*_cnn_v2_e4_{val,test}.json` and the baseline rescore. Last column: mean |log ratio| over channels
(0 = perfect).

| Set | r3_cnn | v2_e4 | mean abs log r |
|---|---|---|---|
| val events | 0.93 / 0.34 / 0.67 / 0.51 / 0.95 / 0.96 | 0.94 / 0.39 / 0.74 / 0.65 / 1.02 / 0.95 | 0.386 -> 0.302 |
| val random | 0.98 / 1.08 / 0.77 / 0.72 / 0.93 / 0.93 | 0.97 / 1.13 / 0.87 / 0.83 / 0.90 / 0.95 | 0.138 -> 0.105 |
| val named | 0.94 / 0.73 / 0.76 / 0.65 / 1.04 / 0.94 | 0.96 / 0.76 / 0.84 / 0.75 / 1.14 / 0.96 | 0.196 -> 0.158 |
| test events | 0.94 / 0.51 / 0.99 / 0.81 / 0.98 / 0.91 | 0.95 / 0.56 / 1.05 / 0.97 / 1.04 / 0.92 | 0.180 -> 0.138 |
| test random | 1.00 / 0.96 / 0.86 / 0.79 / 0.88 / 0.95 | 0.99 / 0.98 / 0.97 / 0.92 / 0.90 / 0.95 | 0.103 -> 0.050 |
| test named | 0.96 / 0.73 / 0.98 / 0.77 / 0.89 / 0.96 | 0.98 / 0.79 / 0.97 / 0.83 / 1.05 / 0.98 | 0.134 -> 0.089 |

Sharpness target met on every set: the winds gain 0.06-0.16, rain 0.02-0.06, no channel overshoots badly (max 1.14,
msl on val named). Blur points were already ~10/10, so this earns nothing on the judge's sheet.

## v2_checks (`runs/v2_checks.csv`)

| | r3_cnn | v2_e4 |
|---|---|---|
| Fani centre error d1 / d2 / d3 (km) | 350 / 245 / 809 | 247 / 153 / 1297 |
| Fani landfall-lat error (deg) | +2.0 | -1.0 |
| Wind-shock west-coast rain change (mm), share negative | -1.29, 92% | -0.80, 75% |
| Edge / interior day-4 RMSE (std) | 0.441 / 0.503 | 0.457 / 0.523 |
| ms per 10-day rollout | 86 | 125 |
| Amphan d2 km / landfall lat (TEST ONLY, not used) | 786 / -7.0 | 623 / -3.0 |

Fani days 1-2 and landfall latitude improve (deeper, better-placed low), day 3 jumps to another minimum in the
search box. Edge and interior error both worse (consistent with the lower ss). ms differs because other
experiments shared the CPU; the in-evaluate number (1 thread) is unchanged at ~135 ms.

## Guards (`V2-WIN`)

- `tests/test_engine.py` (active model) passes; the same contract test pointed at `v2_e4` passes.
- Null test exact: max |zero-perturbation - null| = 0.0.
- Reach probe day-1 rmax 9.67 (val) / 9.9 (test) cells <= 11.
- 10-day rollout 0.10 s on 2 threads (< 1 s).
- Repo scan `scripts/check_repo.py`: 15/15.

## Verdict: LOSES (no extra seeds)

- Target: day-8 variance ratios nearer 1 -> **yes** on all 6 window sets; uncapped ss -> **no, much worse**
  (random 0.138 -> 0.093 val, 0.118 -> 0.074 test).
- Val mean 64.72 vs 67.92 (-3.20) and test mean 63.01 vs 67.96 (-4.95): both far below the -0.2 limit.
- Shocks 12.75 / 12.38 vs 13.12: lower.
- Guards all hold.

Reading: this is the double-penalty trade-off. For a single deterministic forecast, the coherence term
`2 max(P, T)(1 - coh)` charges the full truth power for unpredictable scales whether the model damps them or not, so
the model stops damping them and keeps (mis-placed) detail; RMSE vs truth rises at days 4-8, which is what the
judge's skill score measures. Subich et al. reported CRPS and lagged-ensemble RMSE, not deterministic RMSE, so the
"RMSE unchanged" expectation in the methods report did not transfer. If AMSE is revisited: a weighted mix
(e.g. MSE + small AMSE weight), AMSE only on the bands beyond the predictable scale, or only in late curriculum
stages; all would be new experiments under `V2-P`.

## Errors found

1. **Spec error (missing detail), fixed before coding:** `V2-E4` did not say whether `SHP-1` spread loss and
   `RAIN-2` heavy-rain weighting stay, nor how bands, latitude weighting, rain and checkpoint selection work.
   Spec now concrete (AMSE replaces MSE + `SHP-1` + `SHP-2`; `RAIN-2` separate pointwise term; radial DCT bands;
   `--loss amse` flag; selection unchanged).
2. **Spec error (target, found by the run):** `V2-E4` assumed sharper day-8 fields and better uncapped ss could come
   together from the loss alone; with one deterministic forecast they trade off. Recorded in the spec result line.
3. **Procedure note (neither code nor spec error):** `train_nn` prints the `ep N` line before it writes
   `runs/ckpt/<name>_last.pt`, so "stop as soon as the ep line appears" could kill the run mid-save. The stop here
   waited for the checkpoint file to update first. The write-then-rename in the code means the worst case would be
   losing one epoch, not a broken file.
4. **Note:** epochs 9-12 ran on 2 threads (coordinator instruction after the v1.1 retrain finished; `V2-RES` only
   binds while that retrain runs). Results are not bit-identical to a pure 1-thread rerun.

No code errors found: the default `--loss mse` path is the unchanged baseline loss (smoke test with defaults ran),
and the baseline rescore reproduced `V2-P` exactly.
