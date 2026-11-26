# v2 experiments E2-E8: results summary

Date: 2026-11-27. Spec: `docs/SPEC.md` section 14 (`V2-P`, `V2-WIN`, `V2-E2`..`V2-E8`).
Data: only the provided `india_climate_train.nc` (`CON-6`). Train 2005-2017, choose on 2018-19 (val), check on 2020-21 (test).
Each experiment ran in its own worktree and branch, one seed, CPU only. Full write-ups: `docs/v2/RESULTS_E*.md` on each branch.

**Bottom line: no experiment beats `r3_cnn` under `V2-WIN`, so v1.1 (`final_cnn`, the `r3_cnn` recipe) stays the submission
and E1 (retrain the winner on 2005-2021) is not triggered.** Two follow-ups are running (E2b, E7b); this file is updated when they land.

## Judge points (automatic parts, out of 70) and shocks (out of 15)

| Exp | Change | Val mean (67.92) | Test mean (67.96) | Shocks val / test (13.12) | Target metric | Verdict |
|---|---|---|---|---|---|---|
| E2 | Soft-CSI loss + bias guard | 67.21 (-0.71) | 67.19 (-0.77) | 13.12 / 12.75 | CSI down on 3 of 4 sets | loses |
| E3 | Moisture-flux convergence input + budget penalty | **67.96 (+0.04)** | 67.43 (-0.53) | **13.50** / 13.12 | wind-shock rain -1.84 mm (base -1.29), 83% negative (base 92%); CSI down 5 of 6 | loses (closest) |
| E4 | AMSE loss on 2-D DCT | 64.72 (-3.20) | 63.01 (-4.95) | 12.75 / 12.38 | sharper (test log-var error 0.103 -> 0.050), but uncapped ss 0.118 -> 0.074 | loses |
| E5 | Gamma likelihood for rain amount | 65.55 (-2.37) | 67.29 (-0.67) | **14.25 / 14.62** | CSI 0.098 -> 0.050 (val events) | loses |
| E6 | Zero padding + random crops | 67.92 (0.00) | 65.55 (-2.41) | 13.12 / 12.38 | edge RMSE 0.441 -> 0.445 (worse) | loses |
| E7 | Advect-and-correct (`advcnn`), warm start | 66.87 (-1.05) | 67.38 (-0.58) | 12.00 / 12.38 | Fani d1 350 -> 154 km, landfall lat +2 -> +1; uncapped ss higher on all 6 sets | loses (most promising) |
| E8 | Noise inputs + afCRPS, serve 1 fixed member | 26.19 | 30.89 | 13.12 / 13.12 | RMSE points 0/30 | loses badly |

## Uncapped skill (the judge caps these, so they show headroom)

| Exp | ss test events / random | csi_skill test events / random |
|---|---|---|
| r3_cnn | 0.167 / 0.118 | 0.108 / 0.105 |
| E2 | 0.168 / 0.117 | 0.096 / 0.098 |
| E3 | 0.177 / 0.117 | 0.091 / 0.103 |
| E4 | 0.125 / 0.074 | 0.116 / 0.090 |
| E5 | 0.160 / 0.116 | 0.067 / 0.093 |
| E6 | 0.179 / 0.106 | 0.126 / 0.086 |
| E7 | **0.171 / 0.125** | 0.106 / **0.109** |
| E8 (member 0) | -0.052 / -0.145 | 0.018 / 0.030 |

## v2 checks (`training/v2_checks.py`)

| Model | Fani error d1/d2/d3 km | Fani landfall lat err | Wind shock W-coast rain (share neg) | Edge / interior RMSE |
|---|---|---|---|---|
| r3_cnn | 350 / 245 / 809 | +2.0 | -1.29 mm (92%) | 0.441 / 0.503 |
| E2 | 350 / 755 / 914 | +5.0 | -0.81 (83%) | 0.446 / 0.506 |
| E3 | 247 / 625 / 930 | 0.0 | **-1.84** (83%) | 0.440 / 0.505 |
| E4 | 247 / 153 / 1297 | -1.0 | -0.80 (75%) | 0.457 / 0.523 |
| E5 | 247 / 755 / 914 | +5.0 | -1.09 (92%) | 0.445 / 0.506 |
| E6 | 247 / 430 / 884 | +1.0 | -1.39 (92%) | 0.445 / 0.509 |
| E7 | **154** / 430 / **714** | +1.0 | -1.52 (83%) | 0.445 / 0.506 |
| E8 | 247 / 430 / 884 | +1.0 | -1.00 (83%) | 0.592 / 0.641 |

Amphan (2020, test year) is reported in each branch's results but never used to choose.

## Why they lost (one line each)

- **E2:** the bias guard pulls heavy-rain frequency bias toward 1, but an uncertain forecast scores its best CSI at bias above 1.
- **E3:** stronger, physically right wind-to-rain response, but one seed and 12 windows cannot separate it from noise; test mean -0.53.
- **E4:** for one deterministic forecast, keeping small scales sharp costs RMSE at days 4-8 (double penalty).
- **E5:** gamma likelihood weights each cell's error by about 1/mean, so heavy-rain cells count less, not more.
- **E6:** zero is "normal weather" only for anomaly channels; for static maps, sin/cos and hidden layers it is an arbitrary edge value.
- **E7:** forecasts better and moves cyclones, but warm_ocean "stable" fails more often and day-1 reach is 11.09 (> 11).
- **E8:** a single CRPS-trained member has error = ensemble-mean error + spread, so RMSE points drop to zero.

## Errors found, classified

| Type | Where | What |
|---|---|---|
| Spec error (missing detail) | V2-E2..E8 | Every experiment text lacked concrete flags, weights or formulas; each was fixed in its branch spec before coding |
| Spec error (design) | V2-E2 | Bias guard toward bias = 1 works against CSI; use a band (e.g. 0.8-1.6) or drop it |
| Spec error (reach) | V2-E3 | A 3x3 stencil with diagonals gives day-1 reach 8*sqrt(2) = 11.3 > 11; plus-shaped stencil required |
| Spec error (design) | V2-E3 | Budget penalty using only the start state's convergence cancels out; average start and end |
| Spec error (hypothesis) | V2-E4 | Assumed sharper fields and better uncapped ss both follow from the loss; they trade off |
| Spec error (hypothesis) | V2-E5 | Gamma NLL does not "keep the tail" relative to sqrt-space MSE with 3x heavy weight |
| Spec error (rationale) | V2-E6 | "Zero padding = normal weather outside" holds only for the 18 anomaly channels |
| Spec error (reach budget) | V2-E7 | A 3 cells/day velocity cap does not bound day-1 reach; the correction blocks add to it |
| Spec error (design) | V2-E8 | afCRPS replacing MSE + serving one member cannot keep RMSE points; the research notes' mitigation was not in the spec |
| Spec gap | V2-WIN | "Beyond noise" has no numeric noise level for v2_checks metrics |
| Spec gap | V2-P | "ms per rollout" has no fixed thread count or load; compare speed side by side only |
| Code error (latent, fixed) | E6 branch | `make_input`, edge maps, SHP-8 box assumed the full 32x36 grid; any crop would break |
| Code risk | `train_nn` | `--resume` checks only curriculum and batch, so changed loss flags would silently mix recipes |
| Code risk | `train_nn` | The "ep N" log line prints before the epoch's checkpoint is written; stop only after the file updates |
| Code (missing feature, added) | E7 branch | `--warm-start` (copy tensors by name and shape) |

## Follow-ups (decided autonomously; results appended below when done)

- **E2b**: soft-CSI loss alone, no bias guard (the second setting V2-E2 allows). Branch `v2/e2-softcsi`.
- **E7b**: E7 with a day-1 reach budget of at most 10 cells by construction plus a targeted shock-stability fix, warm-started from `v2_e7`. Branch `v2/e7-advect-correct`.
- Not repeated: E4, E5, E6, E8 (failure is in the method for our scoring, not the setting).
