# v2 experiment E7: one advect-and-correct model (`advcnn`)

Spec: `V2-E7` (protocol `V2-P`, win rule `V2-WIN`, resources `V2-RES`). Branch `v2/e7-advect-correct`.
Model: `artifacts/models/v2_e7.pt` (family `advcnn`, 284,315 parameters). Log: `runs/logs/v2_e7.log`.

## What was built

`twin/nets/advcnn.py` (`AdvectCorrectNet`, registered as family `advcnn`). One 1-day step = 2 substeps:

1. **Velocity.** A small CNN (3x3 conv 64 -> 32, GELU, 1x1 -> 16) on the hidden features predicts a displacement
   (dy, dx) for 8 feature groups: one per anomaly variable (6) and 2 groups of learned channels. The vector length is
   squashed (radial tanh) to < 1.5 cells per substep, so the total displacement is < 3 cells/day.
2. **Transport.** The 32-channel transport set u = [6 anomaly maps, 26 channels lifted by a 1x1 conv from the hidden
   features] is advected semi-Lagrangian (`grid_sample`, bicubic, border padding, `align_corners=True`) and mixed
   with the unadvected u through a learned per-channel weight a = sigmoid(alpha): u' = u + a (adv(u) - u). The
   anomaly part updates z directly, and the whole change is projected back into the hidden features by a 1x1 conv
   without bias.
3. **Correction.** Residual blocks on the hidden features (local CNN). Substep 1 runs block 0, substep 2 runs blocks
   1 and 2.

Output = (z_end - z_last) + out(h)[:6] (change of the anomaly) and out(h)[6] (rain-gate logit), so
`--rain-head two_part` works through the existing rollout unchanged.

At zero velocity the network is exactly r3_cnn (`inp` -> 3 blocks -> `out`). The velocity heads start at zero
output, so the warm-started model starts at r3_cnn's forecast. `tests/test_advcnn.py` checks this, the cap, and that
a Gaussian blob in a constant velocity field moves by the expected cells (3, -2 per day; bilinear and bicubic).

**What was reused from `r3_cnn`** (new `train_nn --warm-start models/r3_cnn.pt`, which copies tensors whose name and
shape match): all 16 tensors, 238,791 of 284,315 parameters (84%): `inp` (3x3, 29 -> 64), `body.0-2`
(3 residual blocks of width 64), `out` (64 -> 7, including the rain-gate row). Newly initialised: velocity heads
`flow.0/1`, lifts `lift.0/1`, projections `proj.0/1` and the mix weights `alpha`.

**Deviation from the brief, deliberate:** the transport channels are lifted by a 1x1 conv from the CNN hidden
features. The hidden features themselves come from r3_cnn's 3x3 input conv, not from a 1x1 conv on the anomaly
state. The raw anomaly maps are carried along in the transport set. This makes the whole r3_cnn reusable and the
starting point exact.

## Training

Baseline recipe (`V2-P`): `python -m training.train_nn --family advcnn --name v2_e7 --rain-head two_part --threads 1
--no-eval --resume --warm-start models/r3_cnn.pt` (curriculum 1:4,2:3,4:3,8:2, sqrt rain, 3x heavy, spread +
spectral, train 2005-2017). Epochs 1-6 ran on 1 thread (306-672 s/epoch). After the v1.1 retrain freed the CPU,
the coordinator asked for a restart: the run was stopped once epoch 6's state had been saved and resumed with
`--threads 3` (the checkpoint's `train_args` therefore shows threads=3). Epochs 7-12 took 320-1017 s. No epoch came
near 40 min, so the width was not reduced. The best validation loss in the last stage was epoch 11 (0.7864).
Smoke test (width 16, 1:1,2:1) loaded through `twin.engine.Engine`, zero == null perturbation exactly; smoke files
deleted.

## Results vs baseline `r3_cnn` (`V2-P` numbers)

All scoring used TWIN_THREADS=1. Leaderboard rows: `v2_e7` (val), `test_v2_e7` (test).

| | r3_cnn | v2_e7 |
|---|---|---|
| val auto events / random / named | 67.51 / 68.12 / 68.12 | 66.60 / 67.00 / 67.00 |
| **val mean** | **67.92** | **66.87 (-1.05)** |
| test auto events / random / named | 68.12 / 67.65 / 68.12 | 67.38 / 67.38 / 67.38 |
| **test mean** | **67.96** | **67.38 (-0.58)** |
| shocks (causality /15) val / test | 13.12 / 13.12 | **12.00 / 12.38** |
| uncapped ss val events / random / named | 0.168 / 0.138 / 0.185 | **0.173 / 0.139 / 0.188** |
| uncapped ss test events / random / named | 0.167 / 0.118 / 0.194 | **0.171 / 0.125 / 0.204** |
| csi_skill val events / random / named | 0.098 / 0.105 / 0.140 | **0.110 / 0.106 / 0.155** |
| csi_skill test events / random / named | 0.108 / 0.105 / 0.162 | 0.106 / **0.109 / 0.159** |
| blur points val events / random | 9.74 / 10.0 | 9.60 / 10.0 |
| reach r95 / rmax day 1 (val) | 2.08 / 9.90 | 4.77 / **11.09** |
| reach r95 / rmax day 1 (test) | 1.88 / 9.67 | 5.02 / 10.75 |
| reach r95 / rmax day 4 (val) | 9.09 / 22.03 | 12.16 / 24.08 |

Shock checks (passes out of 5 event windows; nonzero, localized and null_ok are 5/5 for both models):

| check | r3_cnn val | v2_e7 val | r3_cnn test | v2_e7 test |
|---|---|---|---|---|
| warm_ocean propagates | 5 | 5 | 5 | **3** |
| warm_ocean stable | 5 | **2** | 4 | **3** |
| monsoon_stall propagates | 0 | **1** | 1 | **2** |
| monsoon_stall stable | 5 | 4 | 5 | 5 |

(The r3_cnn shock breakdown was recomputed with `score_shocks` without writing to the leaderboard; its points match
the recorded 13.12.)

`training.v2_checks` (`runs/v2_checks.csv`):

| | r3_cnn | v2_e7 |
|---|---|---|
| **Fani centre error day 1 / 2 / 3 (km)** | 350 / 245 / 809 | **154** / 430 / **714** |
| **Fani landfall-latitude error (deg)** | +2.0 | **+1.0** |
| Amphan error day 2 (km), report only | 786 | 312 |
| Amphan landfall-latitude error, report only | -7.0 | 0.0 |
| monsoon_stall west-coast rain change (mm), share < 0 | -1.29, 92% | -1.52, 83% |
| edge / interior day-4 RMSE (std) | 0.441 / 0.503 | 0.445 / 0.506 |
| ms per 10-day rollout (v2_checks, 1 thread, loaded machine) | 86 | 145 |
| ms per 10-day rollout, 2 threads, side by side now | 71 | 140 |

Extra diagnostics (scratch script, Fani from 1 May 2019): the model now **moves** the low. Its centre goes
16.5N -> 21.5N -> 25.5N against the real 17.5N -> 20.5N -> 24.5N. The latitude is right within 1 degree, but the
track drifts north-north-west where the real storm turned north-east (day-2/3 longitude 82.5-83.5E vs 86.5-90.5E).
That longitude error is what makes day 2 worse in km. The learned msl displacement reaches 2.4 cells/day, under the
3-cell cap. The mix weights stayed near their start (a = 0.44-0.53), so both paths are used.

## Verdict under `V2-WIN`: does NOT win

- Target (Fani track): **mixed.** Day 1 and day 3 errors and the landfall latitude improve, but day 2 gets worse
  (430 vs 245 km). The model moves cyclones (Amphan, test-only, lands within 0 degrees latitude instead of -7), but
  the cyclone track is not better beyond noise on all days.
- val mean -1.05 and test mean -0.58: both lower than the 0.2 tolerance allows. **Fail.**
- Shocks 12.00 / 12.38 vs 13.12. **Fail.**
- Guards: null test exact (0.0), `tests/test_engine.py` contract passes with v2_e7, 10-day rollout 140 ms on
  2 threads (< 1 s), repo scan 15/15. **Reach day-1 rmax 11.09 on val is just over 11. Fail** (10.75 on test).

**Why it loses.** RMSE and CSI points are capped (30/30 and 15/15 on every set for both models). The uncapped skill
is actually higher than r3_cnn on all 6 ss columns and 4 of 6 csi_skill columns. All of the judge-point loss comes
from the shock checks: the warm-ocean shock becomes **unstable** (response > 5 std) in 3 of 5 val windows. A plausible
cause, not proven: advection concentrates the +3 K / +15% tcwv box, and the rain response amplifies it. Monsoon_stall
"propagates" improves (1 vs 0 val, 2 vs 1 test): the transport does what it was meant to do there. The day-1 reach grows
because the per-substep velocity stencil and the 3 residual blocks add reach on top of the 3-cell transport.

**Suggested follow-up (not run, one change per experiment):** E7 + `SHP-8` shock guard, and/or one residual block
per substep (reach budget), and a K=3 substep variant. None of these was run here.

## Errors found

1. **Code (latent hazard), `training/train_nn.py`:** the `ep N` line is printed BEFORE the epoch's state is saved to
   `runs/ckpt/<name>_last.pt`. A procedure that stops training "as soon as the ep line appears" can lose that epoch,
   or catch a half-written `.tmp` (the rename keeps `_last.pt` itself safe). I waited for the checkpoint's timestamp to
   change before stopping. Fix: print after saving, or print a "state saved" line. Not changed here: it is shared
   code across experiments.
2. **Code (missing feature):** `train_nn` had no way to warm-start from another checkpoint, which `V2-E7` requires.
   Added `--warm-start` (copies tensors whose name and shape match; ignored on resume because the resumed state
   overwrites it).
3. **Spec gap, `V2-E7` / `LOC-1`:** "velocity capped <= 3 cells/day" does not bound the day-1 reach. The velocity
   net's stencil and the correction blocks add to it, and this model reached rmax 11.09 > 11. `V2-E7` should state
   a reach budget, e.g. total blocks + stencils such that the theoretical radius <= 11, or require a reach check
   before the full run. The brief's phrase "day-1 response within ~3 cells of the shock box" is the judge's 3 + 8k
   rule (11 cells on day 1), not a 3-cell limit.
4. **Spec gap, `V2-WIN`:** "beyond noise" is undefined for a single-storm, 3-day track metric (Fani). With one case
   a mixed result (better days 1/3, worse day 2) cannot be judged. The spec should name the number, e.g. the mean of
   days 1-3, and a margin.
5. **Spec, `V2-P` speed baseline (86 ms):** timings depend on machine load (r3_cnn measured 71 ms side by side
   today, v2_checks reported 86). Speed comparisons should be side by side. Not a decision issue here (both are far
   under 1 s).

---

# E7b: reach budget + shock guard (fine-tune of v2_e7)

Spec: `V2-E7b` (added before the code; the change log classifies the `V2-E7` reach gap as a spec error).
Model: `artifacts/models/v2_e7b.pt` (153,243 parameters). Log: `runs/logs/v2_e7b.log`.

## Diagnosis on v2_e7 (warm_ocean, val event windows, max |shocked - base| in judge std)

| window | r3_cnn | v2_e7 as is | v2_e7, velocity replayed from the unshocked run | v2_e7, zero velocity |
|---|---|---|---|---|
| monsoon_onset | 4.14 | 5.47 | 5.06 | 8.40 |
| heavy_rain | 4.52 | 6.90 | 6.56 | 6.19 |
| heatwave | 4.60 | 7.03 | 7.02 | 7.66 |
| cyclone | 2.04 | 1.88 | 1.71 | 1.69 |
| winter_disturbance | 0.61 | 0.75 | 0.76 | 0.75 |

The overshoot is always rain (tp), on days 1-6. In the "velocity replayed" run the velocity head cannot react to the
shock, and the overshoot barely changes. With zero velocity (correction path only) it is as bad or worse. So the
velocity head does **not** pile the anomaly up. The fine-tuned **correction/output path amplifies the rain response**
to the warm, moist box. r3_cnn already sits at 4.1-4.6 std, close to the 5 std limit.

**Theoretical day-1 reach of E7** (Euclidean, the probe's metric) = sqrt 2 x (sum of stencil half-sides on the
longest path) + displacement cap:
- input 3x3 conv: 1
- bicubic lookup in each substep: 2 + 2
- block 0: 2
- blocks 1-2: 4
- displacement cap: 3.0 cells/day

That is sqrt 2 x 11 + 3 = **18.6 cells** (13 Chebyshev). The velocity-head 3x3 stencil (half-side 1) is never longer
than the lookup it steers. Measured rmax was 11.09.

## Changes (spec V2-E7b)

- **(a) Reach budget.** Bilinear lookups (half-side 1), cap 2.5 cells/day, blocks 1-2 become 1x1 "reaction" blocks
  (`--net '{"max_disp": 2.5, "interp": "bilinear", "block_k": [3, 1, 1]}'`). Reach = sqrt 2 x (1 + 1 + 2 + 1) + 2.5
  = **9.57 <= 10** by construction. `AdvectCorrectNet.reach_bound()` computes it. The new
  `tests/test_advcnn.py::test_reach_budget` perturbs one input cell under random weights and capped random
  velocities: nothing moves beyond the bound (empirical max 7.6).
- **(b) Stability fix.** `SHP-8` shock guard, `--shock-loss 20` (penalty above 4 judge std, every 4th batch).
- **Warm start** from `models/v2_e7.pt`: all 31 tensors reused. 27 copied by name and shape. The 4 block-1/2 conv
  weights (3x3 -> 1x1) were summed over their taps. The cap and interpolation changes need no new tensors.
- Fine-tune: `--curriculum 4:2,8:2 --lr 3e-4 --threads 3`. Epochs took 266-451 s, 23 min in total. The best last-stage
  validation loss was epoch 4 (0.8153; v2_e7 had 0.7864).

## Results (same table as E7)

| | r3_cnn | v2_e7 | v2_e7b |
|---|---|---|---|
| val auto events / random / named | 67.51 / 68.12 / 68.12 | 66.60 / 67.00 / 67.00 | 65.07 / 66.11 / 67.75 |
| **val mean** | **67.92** | 66.87 | **66.31 (-1.61)** |
| test auto events / random / named | 68.12 / 67.65 / 68.12 | 67.38 / 67.38 / 67.38 | 63.18 / 64.82 / 67.75 |
| **test mean** | **67.96** | 67.38 | **65.25 (-2.71)** |
| shocks val / test | 13.12 / 13.12 | 12.00 / 12.38 | 12.75 / 12.75 |
| uncapped ss val events / random / named | 0.168 / 0.138 / 0.185 | 0.173 / 0.139 / 0.188 | 0.145 / 0.129 / 0.168 |
| uncapped ss test events / random / named | 0.167 / 0.118 / 0.194 | 0.171 / 0.125 / 0.204 | 0.147 / 0.110 / 0.186 |
| csi_skill val events / random / named | 0.098 / 0.105 / 0.140 | 0.110 / 0.106 / 0.155 | 0.083 / 0.089 / 0.123 |
| csi_skill test events / random / named | 0.108 / 0.105 / 0.162 | 0.106 / 0.109 / 0.159 | 0.070 / 0.097 / 0.142 |
| judge CSI points val ev/rnd, test ev/rnd | 15 / 15, 15 / 15 | 15 / 15, 15 / 15 | 12.5 / 13.4, 10.4 / 14.5 |
| reach rmax day 1 val / test | 9.90 / 9.67 | 11.09 / 10.75 | **6.77 / 7.00** |
| reach r95 day 1 val / test | 2.08 / 1.88 | 4.77 / 5.02 | 3.46 / 2.85 |
| warm_ocean stable / propagates (val; test) | 5 / 5; 4 / 5 | 2 / 5; 3 / 3 | 4 / 4; 4 / 5 |
| monsoon_stall propagates (val; test) | 0; 1 | 1; 2 | 1; 0 |
| **Fani error day 1 / 2 / 3 (km)** | 350 / 245 / 809 | 154 / 430 / 714 | **308 / 104 / 528** |
| **Fani landfall-latitude error** | +2.0 | +1.0 | **0.0** |
| Amphan day-2 error / landfall lat (report only) | 786 km / -7.0 | 312 km / 0.0 | 152 km / +1.0 |
| wind-shock west-coast rain (share < 0) | -1.29 mm (92%) | -1.52 mm (83%) | -1.94 mm (92%) |
| edge / interior day-4 RMSE (std) | 0.441 / 0.503 | 0.445 / 0.506 | 0.451 / 0.514 |
| ms per 10-day rollout, 2 threads, side by side | 37 | 75 | 53 |

Guards for v2_e7b: null test exact (0.0), `tests/test_engine.py` contract passes, 53 ms (< 1 s), repo scan 15/15,
reach rmax 6.8 / 7.0 (<= 11). Warm_ocean maxima on val events after the fix: 2.65 / **6.94** / 3.60 / 2.45 / 0.58 std.
The guard fixed 2 of the 3 overshooting windows, but heavy_rain (day 1) still fails.

## Verdict under V2-WIN: does NOT win

- Target: the best Fani track of any model so far. The mean of days 1-3 is 313 km (r3_cnn 468, v2_e7 433), and the
  landfall latitude is exact. Shock propagation is mixed.
- val mean -1.61 and test mean -2.71: **fail**.
- Shocks 12.75 < 13.12: **fail** (better than v2_e7's 12.00 / 12.38).
- All guards hold, including reach, which is now 6.8 / 7.0.

**What it costs:** E7b loses heavy-rain CSI (judge CSI points drop below 15 on 3 of 4 sets) and some uncapped skill.
Two things changed at once, so the cause is confounded:
- the 1x1 reaction blocks remove 4 of the 7 spatial convs;
- the SHP-8 shock guard also cost the U-Net events CSI in phase 2a (0.0615).

**Next step if E7 continues:** separate the two changes.
1. Reach budget only, with the 3x3 capacity kept elsewhere: e.g. 3x3 in block 2 instead of block 0, or bilinear +
   cap 2.0 + one more 3x3. Every variant must still satisfy sqrt 2 x sum + cap <= 10.
2. A rain-only guard: cap the per-step change of the rain amount, or apply the guard only to tp. This targets the
   exact failure found, and avoids a guard that taxes every channel.

## Errors found in E7b

1. **Spec error (fixed), `V2-E7`:** the velocity cap was taken as the locality guarantee. The theoretical reach was
   18.6 cells. `V2-E7b` and the change-log line record the reach-budget arithmetic.
2. **Code (cosmetic), `train_nn --warm-start` log:** the "reused ... params" count sums the source tensor sizes, so it
   prints 284,315 when the 3x3 -> 1x1 collapsed tensors are smaller in the new net (new net: 153,243 params).
3. **Protocol note (spec):** `V2-E7b` changes two things (reach + guard) by design. It cannot say which change cost
   the CSI. A follow-up should change one thing.

---

# E7c: reach budget only (no shock guard), the last v2 follow-up

Spec: `V2-E7c`. Model: `artifacts/models/v2_e7c.pt` (218,779 parameters). Log: `runs/logs/v2_e7c.log`.

**One change from v2_e7, the locality design:** bilinear lookups, cap 2.5 cells/day, blocks 0-1 kept as 3x3
residual blocks, block 2 a 1x1 reaction block. Guarantee: theoretical bound <= 13 and measured rmax day 1 <= 10.

- Arithmetic: sqrt 2 x half + 2.5 <= 13 gives half <= 7.42, so 7 half-sides fit:
  - input 3x3 conv: 1
  - two bilinear lookups: 1 + 1
  - blocks 0 and 1 (two 3x3 convs each): 2 + 2
  - block 2 (1x1): 0
- Bound = sqrt 2 x 7 + 2.5 = **12.40**. `test_reach_budget` now checks both E7b (<= 10) and E7c (<= 13); it passes.
- No `--shock-loss`.
- Warm start from v2_e7: all 31 tensors. Block 2's two conv weights were summed over their taps; everything else was
  copied exactly.
- `--curriculum 4:2,8:2 --lr 3e-4 --threads 6`: 16 minutes (epochs 156-335 s). The best last-stage val loss was
  epoch 4 (0.8027).
- Val and test were scored at the same time (separate processes, TWIN_THREADS=1 each); both leaderboard rows are present.

## Results (all E7 variants)

| | r3_cnn | v2_e7 | v2_e7b | v2_e7c |
|---|---|---|---|---|
| val auto events / random / named | 67.51 / 68.12 / 68.12 | 66.60 / 67.00 / 67.00 | 65.07 / 66.11 / 67.75 | 67.15 / 67.38 / 67.38 |
| **val mean** | **67.92** | 66.87 | 66.31 | **67.30 (-0.62)** |
| test auto events / random / named | 68.12 / 67.65 / 68.12 | 67.38 / 67.38 / 67.38 | 63.18 / 64.82 / 67.75 | 65.42 / 66.03 / 67.38 |
| **test mean** | **67.96** | 67.38 | 65.25 | **66.28 (-1.68)** |
| shocks val / test | 13.12 / 13.12 | 12.00 / 12.38 | 12.75 / 12.75 | 12.38 / 12.38 |
| uncapped ss val events / random / named | .168 / .138 / .185 | .173 / .139 / .188 | .145 / .129 / .168 | .164 / .134 / .175 |
| uncapped ss test | .167 / .118 / .194 | .171 / .125 / .204 | .147 / .110 / .186 | .157 / .115 / .192 |
| csi_skill val | .098 / .105 / .140 | .110 / .106 / .155 | .083 / .089 / .123 | .112 / .109 / .144 |
| csi_skill test | .108 / .105 / .162 | .106 / .109 / .159 | .070 / .097 / .142 | .087 / .105 / .155 |
| judge CSI points val ev/rnd, test ev/rnd | 15/15, 15/15 | 15/15, 15/15 | 12.5/13.4, 10.4/14.5 | 15/15, 13.0/15 |
| reach rmax day 1 val / test | 9.90 / 9.67 | 11.09 / 10.75 | 6.77 / 7.00 | **8.72 / 8.81** |
| theoretical bound (Euclidean cells) | 9.9 | 18.6 | 9.57 | 12.40 |
| warm_ocean stable, val / test (of 5) | 5 / 4 | 2 / 3 | 4 / 4 | 3 / 3 |
| warm_ocean propagates, val / test | 5 / 5 | 5 / 3 | 4 / 5 | 4 / 4 |
| monsoon_stall propagates, val / test | 0 / 1 | 1 / 2 | 1 / 0 | 2 / 1 |
| **Fani error day 1 / 2 / 3 (km)** | 350 / 245 / 809 | 154 / 430 / 714 | 308 / 104 / 528 | 350 / **0** / 607 |
| **Fani landfall-latitude error** | +2.0 | +1.0 | 0.0 | **0.0** |
| Amphan day 2 / landfall lat (report only) | 786 km / -7.0 | 312 km / 0.0 | 152 km / +1.0 | 104 km / 0.0 |
| wind-shock west-coast rain (share < 0) | -1.29 (92%) | -1.52 (83%) | -1.94 (92%) | -1.98 (92%) |
| edge / interior day-4 RMSE | .441 / .503 | .445 / .506 | .451 / .514 | .452 / .512 |
| ms per 10-day rollout, 2 threads, side by side | 37 | 75 | 51 | 60 |

Guards for v2_e7c: null test exact (0.0), `tests/test_engine.py` contract passes, 60 ms (< 1 s), repo scan 15/15,
reach 8.72 / 8.81 (<= 10, the V2-E7c guarantee holds). Warm_ocean maxima on val events: 4.48 / **7.26** / **5.35** /
2.74 / 0.73 std. heavy_rain and heatwave still overshoot, in rain on day 2.

## Verdict under V2-WIN: does NOT win

- Target: the Fani track gain holds. Mean error over days 1-3 is 319 km (r3_cnn 468), the landfall-day centre is
  exact (0 km, latitude error 0.0), and Amphan's landfall latitude is exact.
- val mean -0.62 and test mean -1.68: **fail** (the tolerance is 0.2).
- Shocks 12.38 < 13.12: **fail**.
- All guards hold.

**What E7c separates:**
- **E7b's val CSI loss came from the shock guard plus the extra 1x1 block.** With only the locality change, val CSI is
  back above r3_cnn (.112 / .109 / .144). On test, events CSI (.087) and the uncapped ss are still a little below
  r3_cnn, so the tighter reach costs some skill on the test years.
- **The shock guard helped stability.** Without it, warm_ocean "stable" falls back to 3/5, versus 4/5 in E7b.

The warm_ocean rain overshoot is a property of the fine-tuned correction/output path, and no E7 variant fixed it
without a guard. Across the three E7 runs the trade is consistent: structural transport fixes the cyclone track (Fani
and Amphan landfall latitude exact) but loses 1-2 shock points to rain instability, and so it loses on the judge's
capped totals. r3_cnn stays the submission model under V2-WIN. As agreed, this is the last v2 follow-up. If anyone
revisits E7 later, the evidence points to a rain-only guard (the tp channel) on top of E7c's locality design.

## Errors found in E7c

- None new in code or spec. One process note: running two `training.evaluate` processes in parallel risks a lost
  `runs/leaderboard.csv` row (each process does a read-modify-write). Both rows were checked and are present here.
  **Code (latent):** the CSV append has no lock.
