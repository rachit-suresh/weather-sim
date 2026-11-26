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
