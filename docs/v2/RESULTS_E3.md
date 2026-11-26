# v2 experiment E3: wind-to-rain coupling (moisture-flux convergence input)

Spec: `V2-E3` (made concrete in this branch, see the change log in `docs/SPEC.md`), protocol `V2-P`, win rule `V2-WIN`.
Branch `v2/e3-wind-rain`, model `artifacts/models/v2_e3.pt`, training log `runs/logs/v2_e3.log` (not committed).

## What changed (one thing versus the `r3_cnn` recipe)

- `--flux-conv`: at every rollout step (training and serving, `twin/nets/rollout.py`) the current state (last day of
  the 3-day window) is decoded to u10, v10 (m/s) and tcwv (kg/m2) with the climatology of its own day, and
  C = -div(tcwv V) x 86400 (mm/day) is computed with centred differences over 2 cells (dx = 111.2 km cos(lat),
  dy = 111.2 km, replicate edges, plus-shaped stencil). C / 19.72 (training-set std of C) clamped to +-6 is the 30th
  input map. Stored in the checkpoint as rollout option `flux_conv`, so `twin/models/neural.py` serves it the same way.
- `--budget-loss 0.05` (training only): soft column-budget penalty on R = d(tcwv) + rain - (C(start) + C(end)) / 2,
  penalty 0.05 x lat-weighted mean(((R_pred - R_true) / 17.0)^2) (17.0 mm = training-set std of R_true). On a
  persistence forecast the penalty is about 0.78 vs MSE 1.15, so with w = 0.05 it is ~3-5% of the loss.
- Command: `python -m training.train_nn --family cnn --name v2_e3 --rain-head two_part --threads 1 --no-eval --resume
  --flux-conv --budget-loss 0.05` (epochs 1-8 on 1 thread; restarted from the epoch-8 state with `--threads 2` once the
  v1.1 retrain had finished, per coordinator instruction). Best val loss 0.797 (epoch 11).

## Results versus the baseline `r3_cnn` (numbers from spec `V2-P`)

Judge points (auto /70: events / random / named) and means:

| | val events | val random | val named | **val mean** | test events | test random | test named | **test mean** | shocks val / test |
|---|---|---|---|---|---|---|---|---|---|
| r3_cnn | 67.51 | 68.12 | 68.12 | **67.92** | 68.12 | 67.65 | 68.12 | **67.96** | 13.12 / 13.12 |
| v2_e3 | 66.88 | 68.50 | 68.50 | **67.96** (+0.04) | 66.83 | 67.35 | 68.12 | **67.43** (-0.53) | 13.50 / 13.12 |

Uncapped skill (mean ss, heavy-rain csi_skill):

| | ss val ev / rnd / named | ss test ev / rnd / named | csi_skill val ev / rnd / named | csi_skill test ev / rnd / named |
|---|---|---|---|---|
| r3_cnn | 0.168 / 0.138 / 0.185 | 0.167 / 0.118 / 0.194 | 0.098 / 0.105 / 0.140 | 0.108 / 0.105 / 0.162 |
| v2_e3 | 0.172 / 0.137 / 0.180 | 0.177 / 0.117 / 0.197 | 0.091 / 0.108 / 0.130 | 0.091 / 0.103 / 0.161 |

Points lost on test events are heavy-rain points (csi_skill 0.091 < 0.10 full mark: 13.7/15) and val events the same
(13.6/15); RMSE points are full (30/30) except test random (29.2/30, ss 0.117 vs 0.118 baseline).

Shocks (event windows): all warm_ocean and monsoon_stall responses are nonzero and localized on day 1, null test
exact on all 10 windows. monsoon_stall "propagates" still fails on 4 of 5 windows in both splits (known edge problem,
unchanged). One test window has warm_ocean "stable" = False; the test shock total equals the baseline (13.12).

Reach probe (1-cell shock, cells): val r95/rmax day 1 = 3.17 / **9.93**, day 4 = 7.74 / 21.09; test day 1 = 1.39 /
**9.62**, day 4 = 9.12 / 20.45 (baseline val 2.08 / 9.9, 9.09 / 22.03; test 1.88 / 9.67, 4.79 / 21.11).
rmax_day1 <= 11 holds (`LOC-1`). The plus-shaped stencil can add at most 1 cell along one axis: 7-cell CNN square
-> max Euclidean 10.63, measured 9.93 because far-corner changes stay below the 1e-4 std threshold.

`training.v2_checks`:

| | Fani err km d1 / d2 / d3 | Fani landfall lat err | wind-shock west-coast rain (days 1-3) | share negative | edge / interior day-4 RMSE (std) | ms / 10-day rollout |
|---|---|---|---|---|---|---|
| r3_cnn | 350 / 245 / 809 | +2.0 | -1.289 mm | 0.92 | 0.441 / 0.503 | 86 |
| v2_e3 | 247 / 625 / 930 | 0.0 | **-1.836 mm** | 0.83 | 0.440 / 0.505 | 140 (1 thread, machine shared) |

Amphan (test year, reported only, never used to choose): day-2 error 385 km (baseline 786), landfall lat error -2.0
(baseline -7.0).

Speed: on 2 threads a 10-step rollout takes 98 ms median (max 100 ms), far under 1 s. The extra work per step is one
stencil and one more input map (30 vs 29); the 1-thread ms above were measured while other experiments shared the
CPU, so they are not comparable to the baseline's 86 ms.

Guards: `tests/test_engine.py` passes with v2_e3 (via `TWIN_MODEL_SPEC`) and with the active model; null test exact
(zero perturbation == no perturbation, max diff 0.0, also checked in the smoke test through `twin.engine.Engine`);
deterministic across calls; reach day-1 max 9.93 <= 11; 10-day rollout 98 ms on 2 threads; repo scan 15/15.

## Verdict under `V2-WIN`: does NOT win

1. Target metric not improved beyond noise. Wind-shock rain response: the mean west-coast drop is larger (-1.84 vs
   -1.29 mm, the physically expected sign), but fewer windows have the right sign (10/12 vs 11/12); on 12 windows this
   is not a clear improvement. csi_skill (the second target) is lower on 5 of 6 sets (val events -0.007, val named
   -0.010, test events -0.017; random +0.003 / -0.002).
2. Test auto mean 67.43 is 0.53 below the baseline (limit: 0.2). Val mean is equal (+0.04).
3. Shocks not lower (val 13.50 vs 13.12, test 13.12 = 13.12) and all guards hold, but 1 and 2 already fail.

So E3 gets no extra seeds. Side observations (not decision criteria): Fani day-1 error and landfall latitude improve
(247 km, 0.0), but days 2-3 are worse (625 / 930 km); event-window ss rises slightly on both splits.
Caveat: single seed, and the run switched from 1 to 2 threads at epoch 8 (float summation order changes, so the run is
not bit-reproducible from scratch).

## Errors found

- **Spec error (missing detail), fixed in `V2-E3` before coding:** the spec did not say which climatology decodes the
  current state (the rollout only had target-day climatology), the units, the stencil shape, the normalisation, the
  input count, the checkpoint option or the budget-penalty formula and weight. Now concrete (spec change log
  "v2-e3"). The rollout gets the last input day's climatology as `clim_init` (training and serving pass it).
- **Spec error (the stencil line "widens reach by 1 cell" was incomplete):** a 3 x 3 (diagonal) stencil on top of the
  CNN's 7-cell square gives a Euclidean reach of 8 x sqrt(2) = 11.3 cells > 11 (`LOC-1` limit). Only a plus-shaped
  centred-difference stencil keeps the theoretical bound at 10.63. The spec now requires the plus shape.
- **Spec error (a naive budget penalty is a no-op):** if R uses only the convergence of the start state, C cancels in
  R_pred - R_true (both start from the same state), and the "budget" penalty is just a tcwv + rain error term. The spec
  now uses the mean of the start and end convergence, so predicted winds enter the penalty.
- **Code observation (not a spec violation):** `train_nn --resume` only checks `--curriculum`/`--batch`; resuming with
  different `--budget-loss` or `--threads` is accepted silently (a different `--flux-conv` fails on the weight shapes).
  Also the `ep N` log line is printed before the epoch's state file is written; a restart triggered by that line must
  wait for `runs/ckpt/<name>_last.pt` to be renewed (done here) or it loses the epoch.
- No code errors found in the baseline path: with the flags off the code paths are unchanged (`clim_init` is only
  read by the `flux_conv` option) and the active-model engine test still passes.
