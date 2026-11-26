# Sweep 1: one baseline per model family

Date: 2026-11-26. Machine: laptop CPU (16 threads), 5 jobs in parallel x 3 threads.

## Setup (identical for every neural family)
- Data split: train 2005-2017, validate 2018-2019 (all numbers below), test 2020-2021 untouched.
- Inputs: 29 channels (docs/MODEL_FAMILIES.md), anomaly space, autoregressive 1-day steps.
- Training: 11 epochs = 5 epochs 1-day loss, 3 epochs 2-day unroll, 3 epochs 4-day unroll. AdamW, lr 1e-3
  OneCycle, batch 32, plain latitude-weighted squared error (no rain transform, no blur loss).
- Scoring: `training/evaluate.py` with the organisers' scoring code. "auto /70" = error skill 30 +
  heavy rain 15 + anti-blur 10 + shocks 15 (repo 15 and brief 15 are not simulated).
  "events" = 5 automatically picked extreme windows (like the judge), "random" = 60 random windows.
- Reach = measured spread of a 1-cell shock: r95 = radius holding 95% of the response, max = farthest
  cell that moved at all. The judge's day-1 hard limit is 11 cells from the box (3 + 8 per day).

## Results

Regenerate: `python -m training.report --prefix sweep_`

| model | auto /70 (events) | auto /70 (random) | gate | error skill (events) | heavy-rain skill (events) | blur /10 | shocks /15 | ms / rollout | reach r95 d1 | reach max d1 | reach r95 d4 | reach max d4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fno:sweep_fno | 63.36 | 61.2 | True | 0.1297 | 0.0783 | 10.0 | 11.62 | 130.6 | 10.25 | 24.08 | 15.94 | 24.08 |
| cnn:sweep_cnn | 62.44 | 62.13 | True | 0.1636 | 0.0633 | 9.45 | 13.5 | 114.4 | 2.0 | 9.9 | 3.9 | 21.76 |
| unet:sweep_unet | 60.93 | 64.36 | True | 0.1465 | 0.051 | 9.4 | 13.88 | 106.7 | 1.08 | 12.1 | 4.28 | 23.87 |
| hybrid:sweep_hybrid | 59.86 | 63.47 | True | 0.1378 | 0.0462 | 9.42 | 13.5 | 46.1 | 0.94 | 10.42 | 3.17 | 19.39 |
| physics:sweep_physics | 59.71 | 59.99 | True | 0.1484 | 0.0441 | 9.59 | 13.5 | 91.9 | 2.74 | 7.07 | 5.91 | 14.12 |
| vit:sweep_vit | 57.06 | 62.29 | True | 0.1365 | 0.0323 | 9.46 | 12.75 | 130.5 | 1.67 | 14.21 | 5.44 | 23.37 |
| convgru:sweep_convgru | 58.19 | 61.43 | True | 0.1286 | 0.0408 | 9.32 | 12.75 | 30.7 | 2.19 | 7.07 | 5.0 | 17.57 |
| lgbm:sweep_lgbm | 53.19 | 50.68 | True | 0.1154 | 0.0251 | 9.32 | 11.25 | 589.6 | 1.41 | 1.41 | 2.77 | 5.66 |

Baseline for comparison: damped anomaly persistence = 43.7 (events) / 42.3 (random), reach 0.

## What we learned

1. **Every neural family is far above the baseline** (57-62 vs 44), and every one passes the gate.
2. **Error skill is maxed out for all of them.** All reach skill 0.13-0.16 on the event windows, past the
   0.12 that earns the full 30 points. Lower error is worth nothing more.
3. **Heavy rain is where the points are.** Best heavy-rain skill is 0.063 (CNN), which is 9.5 of 15 points.
   This one component explains most of the gap between families. Decisions D (rain handling) and E
   (blur control) are where the next points come from, not a bigger model.
4. **Shock tests: 12.75-13.9 of 15 for everyone.** The usual miss is the Arabian Sea wind shock not
   "spreading outward" (the box touches the west edge of the grid), which links to decision F (edges).
5. **Reach, measured vs theory.** 95% of a shock's response stays within 1-3 cells on day 1 for every model.
   The effective reach is far below the theoretical one. The farthest cell that moves at all matches theory:
   CNN 9.9 = 7 cells diagonally (7 x sqrt 2), U-Net 12.1, ViT 14.2. ViT and U-Net go past the 11-cell
   day-1 line, but with < 5% of the energy, so they still pass. That is a risk, not a failure.
6. Speed is a non-issue for the networks: 46-130 ms per 10-day forecast while 5 trainings ran on the same CPU
   (limit 5,000 ms). LightGBM is the slowest at 590 ms, still fine.
7. **FNO (the control) scored highest on event windows (63.4), but is the riskiest.** Its response to a 1-cell
   shock spreads over 10 cells (95% radius) on day 1 and touches the whole grid; every other network keeps 95%
   within 1-3 cells. 3 of its 10 shock runs exceeded 5 std ("unstable"), so it has the lowest shock score of the
   networks (11.6/15). It passes "localized" only because the judge's boxes are large (12 x 12 cells); a small
   live shock in the desk review would show the global spread. Its rain skill (0.078) is the best, so the
   lesson is "global context helps rain", which a LOCAL design must get another way (e.g. U-Net level, more steps).
8. **LightGBM (53.2) beats the baseline but trails every network.** Reach is exactly 1 cell/day (1.41 = one
   diagonal), and it was trained on 1-day steps only, so it drifts over 10 days.
9. **Ranking is close and noisy** (5 event windows). CNN, U-Net and the CNN+transformer hybrid are within ~2-4
   points of each other on both window sets; physics hybrid is about 2 points lower but has the most physical
   reach (7.1 cells max on day 1, 14.1 on day 4). All were trained with the plain recipe; decisions B-F will move
   these numbers more than the choice of family.

## Not finished in this session (laptop shutdown)
- ConvGRU finished just before shutdown: 58.2 / 61.4, mid-pack; memory across days did not help with the plain
  recipe. The light graph network (width 48, 4 rounds) did not finish.
  Resume: `bash scripts/resume_sweep.sh <python>`.
- FNO was a control experiment (expected to fail locality).
