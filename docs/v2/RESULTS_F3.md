# V2-F3: rain stability guard (run `v2_f3`)

Date: 2026-11-27. Branch `v2/f-guard`. Spec: `docs/SPEC.md` `V2-F-P`, `V2-F3` (made concrete on this branch before
any code), `V2-WIN`. Data: only `india_climate_train.nc`. One seed (0), CPU.

## What changed (one thing)

Same fine-tune as the F0 control (`--warm-start models/r3_cnn.pt --curriculum 4:2,8:2 --lr 3e-4 --seed 0`, CNN,
two-part rain head, train 2005-2017). The only change is a rain-only hinge guard:
`--shock-loss 20 --shock-channels tp --shock-hinge 3.5 --shock-box warm_ocean`.

- Perturbation = the judge's `make_shock("warm_ocean")`: +3 K t2m and +15% tcwv on the 12 x 12 Bay of Bengal box,
  last input day only.
- Response = shocked minus unshocked training rollout. Rain is decoded to mm/day and divided by the judge's rain std
  (7.553 mm/day, from `artifacts/stats.json`).
- Penalty = `20 x mean(relu(|response_tp| / 7.553 - 3.5)^2)`, added on every 4th batch. It is zero below 3.5 std.
- Check that the training path matches the judge: on 6 training-year windows, the training-path rain maximum fell
  within 0.0-0.3 std of the judge's own number (for example 2.58 vs 2.56, 3.06 vs 3.21, 1.48 vs 1.76).
- How W was chosen: the first guess was W = 3, from E7b's per-rain-cell strength. A check on training windows only
  showed the guard was non-zero in 24 of 24 batches, but with W = 3 its gradient was only 2.3% (unroll 4) and 0.8%
  (unroll 8) of the data gradient. The spec was amended to W = 20, which gives about 10% of the data gradient. No
  val or test number was used to choose W.
- Training: epochs took 665 / 600 / 1215 / 538 s. The guard was non-zero in 31-35 of 38 guard batches and its mean
  fell from 1.1e-4 at the warm start to about 1-2e-5. The machine slept after epoch 3, and the run was resumed from
  `runs/ckpt/v2_f3_last.pt` with `--resume --threads 4`. The best last-stage val loss was epoch 4 (0.7958; F0 0.7962).

## Results

| | r3_cnn (baseline) | F0 `v2_f0` (control) | **F3 `v2_f3`** |
|---|---|---|---|
| val events / random / named | 67.51 / 68.12 / 68.12 | 67.14 / 68.12 / 68.12 | 67.49 / 68.50 / 68.50 |
| **val mean** | **67.92** | 67.79 (-0.13) | **68.16 (+0.24)** |
| test events / random / named | 68.12 / 67.65 / 68.12 | 67.38 / 66.28 / 67.38 | 68.12 / 67.08 / 68.12 |
| **test mean** | **67.96** | 67.01 (-0.95) | **67.77 (-0.19)** |
| shocks val / test (of 15) | 13.12 / 13.12 | 13.12 / 12.38 | **13.50 / 13.12** |
| warm_ocean stable, val / test (of 5) | 5 / 4 | 3 / 3 | **5 / 4** |
| warm_ocean propagates, val / test | 5 / 5 | 5 / 5 | 5 / 5 |
| monsoon_stall propagates, val / test | 0 / 1 | 2 / 0 | 1 / 1 |
| uncapped ss val events / random / named | 0.168 / 0.138 / 0.185 | 0.171 / 0.136 / 0.190 | 0.171 / 0.136 / 0.189 |
| uncapped ss test events / random / named | 0.167 / 0.118 / 0.194 | 0.161 / 0.116 / 0.198 | 0.162 / 0.116 / 0.198 |
| csi_skill val events / random / named | 0.098 / 0.105 / 0.140 | 0.095 / 0.108 / 0.142 | 0.095 / 0.107 / 0.141 |
| csi_skill test events / random / named | 0.108 / 0.105 / 0.162 | 0.111 / 0.105 / 0.165 | 0.113 / 0.105 / 0.163 |
| Fani error d1/d2/d3 km, landfall lat | 350 / 245 / 809, +2.0 | 350 / 245 / 809, +2.0 | 350 / 245 / 809, +2.0 |
| wind-shock W-coast rain (share < 0) | -1.29 mm (92%) | -1.24 mm (83%) | -1.25 mm (83%) |
| edge / interior RMSE (std) | 0.441 / 0.503 | 0.444 / 0.504 | 0.444 / 0.503 |
| reach rmax day 1, val / test (<= 11) | 9.90 / 9.67 | 9.90 / 9.67 | 9.90 / 9.67 |
| ms per rollout (evaluate) | 80-86 | 83 | 46-49 |

Amphan (test year, report only): 786 km on day 2 and landfall latitude -7 for all three models.

### Warm_ocean worst-case change per event window (max |shocked - base| over channels, cells and 10 days, judge std; limit 5)

| window | r3_cnn val | F0 val | **F3 val** | r3_cnn test | F0 test | **F3 test** |
|---|---|---|---|---|---|---|
| monsoon_onset | 4.14 | 4.25 | **3.05** | 2.13 | 2.73 | 2.24 |
| heavy_rain | 4.52 | 5.15 | **3.93** | 3.96 | 5.25 | **3.08** |
| heatwave | 4.60 | 5.41 | **3.50** | 1.76 | 1.93 | 1.71 |
| cyclone | 2.04 | 2.41 | 1.99 | 8.24 | 8.29 | **6.59** |
| winter_disturbance | 0.61 | 0.60 | 0.58 | 1.05 | 1.03 | 1.04 |
| **worst** | 4.60 | 5.41 | **3.93** | 8.24 | 8.29 | **6.59** |

Every maximum is rain (tp), on days 1-6, except winter_disturbance on val, where it is u10.

### Guards (`V2-WIN`)

- `tests/test_engine.py` passes with `active.json` pointed at `models/v2_f3.pt`. The engine reported `cnn:v2_f3`,
  and `active.json` was restored to r3_cnn afterwards.
- The null test is exact (max |null - base| = 0.0 std over all 10 event windows).
- Reach rmax on day 1 is 9.90 / 9.67, within the 11-cell limit.
- A 10-day rollout takes 53 ms on 2 threads (60 ms max), well under 1 s.
- `scripts/check_repo.py` scores 15/15.

## Verdict

**F3 passes `V2-WIN` against r3_cnn, narrowly, and beats the F0 control on its target metric.**

- Against r3_cnn:
  - The val mean is +0.24.
  - The test mean is -0.19, inside the 0.2 tolerance by 0.01.
  - Shocks are 13.50 / 13.12, not lower.
  - All guards hold.
  - Target metric: the warm_ocean worst case fell from 4.60 to 3.93 on val and from 8.24 to 6.59 on test. It is lower
    in 9 of the 10 windows. The exception is test monsoon_onset (2.13 to 2.24, far below the limit).
  - csi_skill did not fall: it moved by -0.003 to +0.005.
- Against F0: F0 shows that fine-tuning alone makes the rain overshoot worse. Its stable count drops to 3/5 on both
  val and test, and its worst case rises to 5.41 / 8.29. F3 keeps r3_cnn's 5/5 and 4/5, and has the lowest worst
  case of the three models in 8 of 10 windows. The exceptions are test monsoon_onset (r3_cnn 2.13) and test
  winter_disturbance (F0 1.03 vs 1.04). F3 is also 0.37 / 0.76 above F0 on the val /
  test means.
- Limits:
  - One seed.
  - The stable *count* is no better than r3_cnn's. The test cyclone window still fails (6.59 > 5, rain on day 1).
  - The +0.38 val shock points come from one more monsoon_stall "propagates" pass, not from warm_ocean, so they are
    probably noise.
  - `V2-F-P` says winners get 3 seeds before anything is combined. F3 is a candidate for that step and for the
    combination with other F winners. It does not replace the submission on its own.

## Errors found, classified

| Type | Where | What |
|---|---|---|
| Spec gap (fixed) | `SHP-8` / `V2-F3` | SHP-8 shocked a random 12 x 12 box. It also took tcwv from the next target day's climatology instead of the judge's last input day. F3 uses the judge's box and day; the defaults are unchanged. |
| Spec error (design, fixed before training) | `V2-F3` | W = 3 was set from per-cell strength. Only a few cells exceed the hinge, so its gradient was 1-2% of the data gradient, which makes the guard a near no-op. W = 20 now follows a gradient-share rule measured on training windows. |
| Spec gap | `V2-WIN` | "Beyond noise" has no number. The test mean (-0.19) sits 0.01 inside the tolerance on one seed. |
| Code risk (minor, not fixed) | training vs engine path | On 2 of 6 checked windows the training-path rain response is 0.15-0.3 std below the judge's. The likely causes are the engine's input clamp, output cleaning or the year-end day-of-year wrap. The guard therefore under-estimates slightly, which errs toward allowing overshoot. |
| Process | run | The machine slept after epoch 3. `--resume` restored the optimiser, scheduler and RNG states but ran at 4 threads instead of 2. The numerics may differ, but the recipe is the same. |
| Code error | none found | The new flags default to the old `SHP-8` path. The random-box branch makes the same RNG calls as before. |
