# Round F, runs F0 / F1 / F2: control, cyclone oversampling, heavy-rain oversampling

Date: 2026-11-27. Branch `v2/f-sample` (from `v2/research` 6f8861c). Spec: `docs/SPEC.md` `V2-F-P`, `V2-F0`, `V2-F1`,
`V2-F2`, and the new `V2-F-S`. I wrote `V2-F-S` (flags, thresholds, sampler, bias guard) and committed it before any code.
Data: only `india_climate_train.nc`. Train 2005-2017, val 2018-19, test 2020-21. CPU, 2 threads per run, one seed (0).

**Bottom line: none of the three runs wins.** F0, the control, is already below `r3_cnn` on test (mean -0.95), so just
fine-tuning longer does not help. F1 and F2 fail `V2-WIN`, and each is also worse than F0 on its own target metric.

## Setup

- All three runs use `train_nn --family cnn --rain-head two_part --warm-start models/r3_cnn.pt --curriculum 4:2,8:2
  --lr 3e-4 --seed 0 --threads 2`. All 16 of 16 tensors were reused. The runs were launched in parallel.
  Logs: `runs/logs/v2_f{0,1,2}.log`. Scoring logs: `runs/logs/v2_f*_score.log`.
- F1 adds `--oversample cyclone --oversample-factor 4`. F2 adds `--oversample heavy_rain --oversample-factor 4`.
- Sampler (`V2-F-S`, code in `training/oversample.py`): each epoch draws as many window starts as before, with
  replacement. A flagged window is 4x as likely to be drawn as an unflagged one. A window is flagged if any of its
  target days is flagged.
  - Cyclone day: the minimum msl anomaly over Bay of Bengal / Arabian Sea sea cells south of 23N, measured against a
    2005-2017 climatology, is below **-8 hPa**. That flags **3.1% of 2005-2017 days**. Flagged windows: 6.9% at unroll 4
    and 11.4% at unroll 8, which become 22.8% and 34% of draws. The flagged days match the known storms (Nargis, Aila,
    Phailin, Hudhud, Vardah, Ockhi and others) plus deep monsoon depressions.
  - Heavy-rain day: the number of cells above 10 mm/day is at or above **309**, the 90th percentile of 2005-2017 days.
    That flags 10.0% of days. Flagged windows: 17.5% at unroll 4 and 22.5% at unroll 8, which become 46% and about 54%
    of draws.
- Best val MSE (unroll 8): F0 0.7962, F1 0.8103, F2 0.8042. The oversampled runs fit the normal validation
  distribution worse.
- Smoke test: `1:1,2:1` with heavy-rain sampling. It loaded through `Engine`, the null test was exact, and
  `--resume` with a different `--oversample` was refused. The smoke files were deleted.

## Results

Judge auto points out of 70, for the window sets events / random / named.

| | r3_cnn | F0 | F1 cyclone | F2 heavy rain |
|---|---|---|---|---|
| val e / r / n | 67.51 / 68.12 / 68.12 | 67.14 / 68.12 / 68.12 | 66.87 / 67.00 / 67.00 | 66.50 / 67.38 / 67.38 |
| **val mean** | 67.92 | 67.79 (-0.13) | 66.96 (-0.96) | 67.09 (-0.83) |
| test e / r / n | 68.12 / 67.65 / 68.12 | 67.38 / 66.28 / 67.38 | 68.50 / 65.41 / 68.50 | 67.75 / 65.73 / 67.75 |
| **test mean** | 67.96 | 67.01 (-0.95) | 67.47 (-0.49) | 67.08 (-0.88) |
| shocks val / test (of 15) | 13.12 / 13.12 | 13.12 / 12.38 | 12.00 / 13.50 | 12.38 / 12.75 |
| ss val e / r / n | 0.168 / 0.138 / 0.185 | 0.171 / 0.136 / 0.190 | 0.166 / 0.130 / 0.178 | 0.168 / 0.133 / 0.178 |
| ss test e / r / n | 0.167 / 0.118 / 0.194 | 0.161 / 0.116 / 0.198 | 0.153 / 0.108 / 0.196 | 0.165 / 0.112 / 0.192 |
| csi_skill val e / r / n | 0.098 / 0.105 / 0.140 | 0.095 / 0.108 / 0.142 | **0.114** / 0.104 / 0.131 | 0.095 / 0.107 / 0.130 |
| csi_skill test e / r / n | 0.108 / 0.105 / 0.162 | 0.111 / 0.105 / 0.165 | 0.114 / 0.099 / 0.150 | 0.110 / 0.101 / 0.154 |
| heavy freq bias val d1/d4/d8 | 1.09 / 1.17 / 1.36 | 1.11 / 1.21 / 1.37 | 1.10 / 1.19 / 1.38 | 1.11 / 1.23 / 1.39 |
| heavy freq bias test d1/d4/d8 | 1.11 / 1.35 / 1.52 | 1.13 / 1.39 / 1.50 | 1.12 / 1.36 / 1.50 | 1.13 / **1.43** / 1.52 |
| raw CSI >10 mm val d1/d4/d8 | 0.552 / 0.422 / 0.361 | 0.554 / 0.426 / 0.358 | 0.553 / 0.417 / 0.354 | 0.551 / 0.426 / 0.360 |
| raw CSI >10 mm test d1/d4/d8 | 0.535 / 0.398 / 0.299 | 0.533 / 0.397 / 0.298 | 0.530 / 0.392 / 0.294 | 0.533 / 0.392 / 0.298 |
| Fani err d1 / d2 / d3 km | 350 / 245 / 809 | 350 / 245 / 809 | 350 / 633 / 1076 | 350 / 245 / **516** |
| Fani landfall lat err | +2.0 | +2.0 | **+1.0** | +2.0 |
| wind-shock W-coast rain (share < 0) | -1.29 mm (92%) | -1.24 (83%) | -1.37 (92%) | -1.21 (83%) |
| edge / interior RMSE (std) | 0.441 / 0.503 | 0.444 / 0.504 | 0.451 / 0.511 | 0.446 / 0.504 |
| reach rmax day 1 val / test | 9.90 / 9.67 | 9.90 / 9.67 | 9.90 / 9.90 | 9.90 / 9.90 |
| Amphan d2 km / lat err (report only) | 786 / -7.0 | 786 / -7.0 | 786 / -7.0 | 786 / -7.0 |

- Frequency bias and raw CSI come from the new `training/heavy_bias.py`: the 60 random windows per split,
  pooled counts. It reproduces the E2 diagnostic for `r3_cnn` exactly. Rows are in `runs/heavy_bias.csv`.
- Guards held for all three models:
  - null test exact (max |diff| 0.0);
  - `tests/test_engine.py` passes, run with `TWIN_MODEL_SPEC` pointing at each checkpoint; `artifacts/active.json`
    was not changed;
  - reach day-1 rmax is at most 9.90 cells;
  - a 10-day rollout on 2 threads takes at most 0.07 / 0.09 / 0.08 s;
  - `scripts/check_repo.py` gives 15/15.

## Verdicts

- **F0 (control): loses to r3_cnn.**
  - Val mean is -0.13, inside the 0.2 margin.
  - Test mean is -0.95, test shocks fall from 13.12 to 12.38, and test random ss falls from 0.118 to 0.116.
  - Four more epochs with a fresh OneCycle at lr 3e-4 do not improve the model. The F0 bar is already below the
    baseline.
- **F1 (cyclone oversampling): loses.**
  - `V2-WIN` fails: val mean -0.96 and val shocks 12.00.
  - Target vs F0: Fani d1 is the same (350 km). d2 and d3 are much worse (633 / 1076 vs 245 / 809 km). Landfall
    latitude improves from +2 to +1. Net, the track is worse.
  - Val-events csi_skill rises (0.114), but random and named are lower, and test random ss drops to 0.108.
- **F2 (heavy-rain oversampling): loses.**
  - `V2-WIN` fails: val -0.83, test -0.88, and shocks are lower on both splits.
  - Target vs F0: csi_skill is equal on val events and lower on the other three sets (val random 0.107 vs 0.108,
    test 0.110 / 0.101 vs 0.111 / 0.105).
  - The bias stays inside 0.8-1.6 on all 6 values, so the guard passes. Test d4 over-forecast still rises
    (1.39 to 1.43) while raw CSI does not.
  - Side effect worth noting: Fani d3 improves to 516 km.

## Errors found, classified

| Type | Where | What |
|---|---|---|
| Spec error (missing detail), fixed | `V2-F1`, `V2-F2` | No flags, region, threshold value, window rule, sampler or bias measurement were defined. Fixed in `V2-F-S` before coding |
| Spec error (design) | `V2-F-P` | The control F0 itself loses to `r3_cnn` on test (-0.95). "Beats F0" then compares against a weaker model, and any F run must first undo the fine-tune's own loss. A fine-tune round should check the control first, or use a gentler schedule (lower lr, no fresh warm-up) |
| Spec error (design) | `V2-F-S` / `V2-F1`, `V2-F2` | "Flagged if any target day is flagged" at unroll 8 turns a 3% / 10% day rate into 11% / 22% of windows and 34% / 54% of draws. That is far more than "4x the rare case". The training distribution shifts and val MSE gets worse. A per-day loss weight, or flagging by the share of flagged target days, would target the rare days more cleanly |
| Spec gap | `V2-WIN` | One seed and one storm (Fani) make it impossible to separate F1's mixed track result (d2/d3 worse, landfall better) from noise |
| Code error | none | Implementation matches `V2-F-S`. With `--oversample none` (F0), the training path is unchanged |
| Code risk, mitigated | `train_nn --resume` | Now also refuses a changed `--oversample` / `--oversample-factor` |
| Process issue (not code) | scoring | Another agent ran this branch's scoring script, because the agents share a scratchpad folder and both used the file names `score.sh` and `guards.py`. That run added duplicate F0/F1 rows (same numbers, different ms). They were deduped: one row per tag, from the original scoring run. `runs/leaderboard.csv` still has no file lock |
