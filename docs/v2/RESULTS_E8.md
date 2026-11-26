# V2-E8 results: fixed-seed CRPS noise input (branch `v2/e8-crps`)

**Verdict: LOSES (V2-WIN fails clearly).** The single served member falls from 67.9 / 68.0 (val / test auto mean) to
26.2 / 30.9. The mean of 2 fixed members does better but still falls far short at 44.0 / 47.7. The target metrics also
get worse: csi_skill drops and rain blur moves toward the 2.5 cliff. Determinism and locality hold.

## What was run

- Spec: `V2-E8` made concrete first (`docs/SPEC.md` section 14 + change-log line), then code.
- Model `models/v2_e8.pt` = `r3_cnn` recipe (CNN 64 x 3 blocks, `RAIN-3` two-part rain, sqrt rain, 3x heavy weight,
  spread + spectral loss, curriculum 1:4,2:3,4:3,8:2, train 2005-2017, seed 0) + ONE change:
  - 2 per-pixel N(0,1) noise input maps appended to the 29 inputs (no pooling or normalisation, so locality is unchanged);
  - M = 2 members per sample, each with fresh noise per step. The squared-error term is replaced by afCRPS
    (alpha 0.95) per channel on anomalies (lat weights, 3x at heavy-rain cells). SHP-1, SHP-2 and the RAIN-3 BCE
    are computed per member;
  - serving uses member 0, with noise `default_rng([0, 0, k])` for step k. It is built once at load time,
    independent of the input. `members: 2` in the model spec serves the mean of members 0 and 1.
- Command: `train_nn --family cnn --name v2_e8 --rain-head two_part --no-eval --resume --noise-maps 2 --crps-members 2`.
  Epochs 1-6 ran on 1 thread. On the coordinator's instruction, after the v1.1 retrain finished, training was
  stopped right after epoch 6's state was saved. It resumed with `--threads 3` (same flags). Epoch times: about
  355 s at unroll 1 and 650-800 s at unroll 2 on 1 thread; 350 / 640-710 / 870-1050 s at unroll 2 / 4 / 8 on
  3 threads. The best epoch was 11 (val MSE of the served member 1.2635). M = 2 was used from the start, as the
  task allowed; no epoch took more than 40 min.
- Scoring: `TWIN_THREADS=1`, `training.evaluate` val (tag `v2_e8`) and test (`test_v2_e8`), `training.v2_checks`
  (`v2_e8`). Second spec, the mean of 2 fixed members: tags `v2_e8_mean2`, `test_v2_e8_mean2`, v2_checks `v2_e8_mean2`.
  Baseline `r3_cnn` was re-scored with `write=False` to get blur ratios. Its numbers matched the spec exactly.

## Judge points (auto /70) per window set

| Model | val events | val random | val named | **val mean** | test events | test random | test named | **test mean** | shocks |
|---|---|---|---|---|---|---|---|---|---|
| r3_cnn (baseline) | 67.51 | 68.12 | 68.12 | **67.92** | 68.12 | 67.65 | 68.12 | **67.96** | 13.12 |
| v2_e8 (member 0, served) | 24.24 | 25.85 | 28.47 | **26.19** | 25.80 | 27.66 | 39.21 | **30.89** | 13.12 |
| v2_e8 mean of 2 members | 44.81 | 33.74 | 53.51 | **44.02** | 46.63 | 31.89 | 64.66 | **47.73** | 12.75 |

Points split (RMSE / CSI / blur), val random: baseline 30 / 15 / 10; member 0: 0 / 2.7 / 10; mean2: 1.1 / 9.9 / 10.

## Uncapped numbers

| | ss val ev / rnd / named | ss test ev / rnd / named | csi_skill val ev / rnd / named | csi_skill test ev / rnd / named |
|---|---|---|---|---|
| r3_cnn | 0.168 / 0.138 / 0.185 | 0.167 / 0.118 / 0.194 | 0.098 / 0.105 / 0.140 | 0.108 / 0.105 / 0.162 |
| v2_e8 member 0 | -0.071 / -0.136 / -0.053 | -0.052 / -0.145 / 0.017 | 0.007 / 0.018 / 0.036 | 0.018 / 0.030 / 0.079 |
| v2_e8 mean2 | 0.047 / 0.004 / 0.069 | 0.060 / -0.008 / 0.108 | 0.069 / 0.066 / 0.090 | 0.060 / 0.061 / 0.140 |

Day-8 blur ratios (pred variance / real; order t2m, tp, u10, v10, msl, tcwv), random windows:

| | val | test |
|---|---|---|
| r3_cnn | 0.98 1.08 0.77 0.72 0.93 0.93 | 1.00 0.96 0.86 0.79 0.88 0.95 |
| v2_e8 member 0 | 1.02 **2.05** 1.00 1.09 1.28 1.08 | 1.03 **2.31** 1.11 1.15 1.22 1.11 |
| v2_e8 mean2 | 1.02 1.43 0.80 0.85 1.16 0.99 | 1.03 1.34 0.89 0.92 1.10 1.01 |

The member is sharp (wind ratios near 1). But rain variance is 2.0-2.3x the real value, close to the judge's
2.5 zero-points cliff. On events (val) the ratios are 0.95 0.59 0.92 0.78 1.30 1.09 (baseline 0.93 0.34 0.67 0.51 0.95 0.97).

Skill per lead (30 val random windows, diagnostic, SS day 1 / 4 / 8):
r3_cnn 0.193 / 0.024 / -0.020; member 0 -0.006 / -0.380 / -0.375; mean2 0.098 / -0.176 / -0.192;
mean of 8 members 0.158 / -0.023 / -0.056. Even the 8-member mean stays below the baseline. The single member
is near zero skill already at day 1.

## Shocks, null test, reach, guards

- Causality points: 13.12 (val and test) for member 0, equal to the baseline. Mean2 gets 12.75 (lower).
  Same total, different pattern. Member 0 val: warm_ocean "stable" fails on 2 of 5 windows and monsoon_stall
  "propagates" fails on 3 of 5. Baseline val: all warm_ocean checks pass and monsoon_stall "propagates" fails on
  5 of 5. So the noisy member passes the Arabian Sea "propagates" check more often, but it is less stable under
  the warm-ocean shock.
- **null_ok = True on every event window, in all 4 evaluations** (val, test, member 0, mean2).
- Determinism proof (smoke and final model, `Engine({"model":"neural","checkpoint":...})`): two calls with the same
  input are bitwise identical. Zero perturbation equals None bitwise (max |diff| 0.0). A fresh Engine, after other
  RNG use, gives bitwise the same output. A 3-step call equals the first 3 steps of a 10-step call. The same
  holds for `members: 2`. Member 0 and the mean of 2 differ (max 486 Pa in msl), so the noise is really used.
- Reach probe: day-1 rmax 9.45 cells (val and test) <= 11 (baseline 9.9 / 9.67). r95 day 1 is 3.82 / 2.69
  (baseline 2.08 / 1.88): the member's response is less concentrated near the shocked cell.
- `tests/test_engine.py` passes with the E8 model active (`TWIN_MODEL_SPEC`). Repo scan: 15/15.
- Speed: 10-day rollout on 2 threads 54 ms (member 0), 81 ms (mean2), r3_cnn 55 ms.
  ms in evaluate (1 thread): 70-84 ms.

## v2_checks (`runs/v2_checks.csv`)

| | Fani err km d1/d2/d3 | landfall-lat err | wind-shock W-coast rain | share < 0 | edge / interior RMSE (std) | ms |
|---|---|---|---|---|---|---|
| r3_cnn | 350 / 245 / 809 | +2.0 | -1.29 mm | 0.92 | 0.441 / 0.503 | 86 |
| v2_e8 member 0 | 247 / 430 / 884 | +1.0 | -1.00 mm | 0.83 | 0.592 / 0.641 | 69 |
| v2_e8 mean2 | 247 / 828 / 714 | +5.0 | -1.07 mm | 0.83 | 0.513 / 0.560 | 126 |

Amphan (test year, report only): d2 error 676 km and landfall-lat error -6.0 for both, vs 786 km / -7.0 for the baseline.

## V2-WIN judgement

| Condition | Result |
|---|---|
| Target (sharpness, csi_skill) improves beyond noise | Sharpness: yes in variance terms, but rain over-shoots (2.0-2.3). csi_skill: **no**, it falls from 0.105 to 0.018 (val random) |
| val / test auto means not lower than baseline by > 0.2 | **No**: -41.7 / -37.1 (member 0); -23.9 / -20.2 (mean2) |
| shocks not lower | Member 0 equal (13.12); mean2 lower (12.75) |
| guards (engine test, null test, reach <= 11, < 1 s, scan 15/15) | All hold |

**LOSES.** No extra seeds (V2-WIN: only winners get them). `r3_cnn` stays.

## Errors found

1. **Spec error (design, `V2-E8`):** "train with afCRPS, serve one fixed member" cannot keep the RMSE points.
   The research notes predicted this as the single-member double penalty: a calibrated member's expected MSE is
   the ensemble-mean MSE plus the ensemble spread. Measured: member-0 SS is already about 0 at day 1 and -0.38 at
   days 4-8. The research note's mitigation ("MSE kept as the primary loss") was not in the spec, and the
   concrete V2-E8 text (and this run) replaced the squared error entirely. If E8 is retried, the spec must first
   change the objective, for example MSE primary + a small afCRPS weight, or a CRPS residual head on top of the
   frozen `r3_cnn` mean. It would also need a rain-variance cap, since the member's day-8 rain variance is
   2.0-2.3x the real value, near the judge's 2.5 cliff.
2. **Spec error (incomplete `V2-E8`), fixed before coding:** the text did not say where the noise enters, which
   loss terms CRPS replaces, how the fixed noise is drawn, which member is served or how the best epoch is chosen.
   It was made concrete in `docs/SPEC.md` (change-log line on the `v2/e8` branch).
3. **Code errors:** none found. The determinism / null proofs pass bitwise. The baseline re-score reproduces the
   spec's r3_cnn numbers exactly. Old checkpoints load unchanged (`noise` is optional in `rollout` config).
4. **Process note (not an error):** epochs 7-12 ran with 3 threads instead of 1, at the coordinator's request after
   the v1.1 retrain freed the CPU. Resuming restores the RNG and optimiser state; only float summation order may differ.
