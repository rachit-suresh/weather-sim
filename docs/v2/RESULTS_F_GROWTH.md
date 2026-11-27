# Fine-tune round F: F4 (unroll noise) and F5 (10-day unroll)

Date: 2026-11-27. Branch `v2/f-growth`. Spec: `docs/SPEC.md` 14b (`V2-F-P`, `V2-F4`, `V2-F5`, new `V2-F-M`), `V2-WIN`.
Data: only `india_climate_train.nc` (`CON-6`). Both runs: `--family cnn --rain-head two_part --warm-start models/r3_cnn.pt
--lr 3e-4 --seed 0 --threads 2`, train 2005-2017, best val loss in the last stage. Control F0 (no change) from branch
`v2/f-sample` (`v2_f0`, same schedule `4:2,8:2`).

- **F4 `v2_f4`:** `--curriculum 4:2,8:2 --unroll-noise 0.1`. Best epoch 4 (val loss 0.7969; F0 0.7962).
- **F5 `v2_f5`:** `--curriculum 4:2,10:2`. Best epoch 3 (10-day val loss 0.8467).

**Why sigma = 0.1.** Each anomaly channel has std 1 in the network's space (`REP-1`), so sigma is the fraction of the
per-channel anomaly std. Measured on 200 training windows, r3_cnn's day-1 error is 0.52 std (0.39-0.69 by channel).
With sigma 0.1 the noise is about 1% of the anomaly variance and about 4% of the day-1 error variance: large enough that
the network must learn to remove it, small enough not to bury the signal or push the model to blur. 0.05 (0.25% /
1% of variance) seemed too small to matter in 4 fine-tune epochs. The smoke test showed the residual step
`z = last + delta` carries the noise forward 1:1 (output RMS change 0.09-0.10 at step 2), which corrected the spec's
first reasoning that convolutions would average the noise away.

**Smoke test (passed, files deleted):** `--unroll-noise 0` gives bit-identical weights to no flag after a real
unroll-2 epoch, and draws no random numbers. Noise changes steps 2+ only, gradients flow, the rain floor holds,
`mode="infer"` ignores it, the checkpoint's `rollout` config has no noise key, serving is deterministic, and
zero perturbation equals null (max diff 0.0).

**Time:** at unroll 4 an epoch took 444-544 s. At unroll 8 (F4) it took 877-977 s, and at unroll 10 (F5) 1122-1227 s.
That is about 1.25x for 1.25x the steps (2 threads, shared CPU with 2 other agents' trainings). Memory was not a
problem (~15 GB free throughout).

## Judge points (automatic parts /70) and uncapped skill

| Model | Val events / random / named | Val mean | Test events / random / named | Test mean | Shocks val / test |
|---|---|---|---|---|---|
| r3_cnn (baseline) | 67.51 / 68.12 / 68.12 | 67.92 | 68.12 / 67.65 / 68.12 | 67.96 | 13.12 / 13.12 |
| F0 `v2_f0` (control) | 67.14 / 68.12 / 68.12 | 67.79 | 67.38 / 66.28 / 67.38 | 67.01 | 13.12 / 12.38 |
| F4 `v2_f4` | 67.15 / 68.12 / 68.12 | 67.80 (-0.12) | 67.75 / 66.71 / 67.75 | 67.40 (-0.56) | 13.12 / 12.75 |
| F5 `v2_f5` | 65.87 / 67.58 / 68.12 | 67.19 (-0.73) | 67.75 / 66.99 / 67.75 | 67.50 (-0.46) | 13.12 / 12.75 |

| Model | ss val ev / rnd / named | ss test ev / rnd / named | csi_skill val ev / rnd / named | csi_skill test ev / rnd / named |
|---|---|---|---|---|
| r3_cnn | 0.168 / 0.138 / 0.185 | 0.167 / 0.118 / 0.194 | 0.098 / 0.105 / 0.140 | 0.108 / 0.105 / 0.162 |
| F0 | 0.171 / 0.136 / 0.190 | 0.161 / 0.116 / 0.198 | 0.095 / 0.108 / 0.142 | 0.111 / 0.105 / 0.165 |
| F4 | 0.173 / 0.136 / 0.190 | 0.161 / 0.116 / 0.198 | 0.095 / 0.107 / 0.142 | 0.111 / 0.105 / 0.166 |
| F5 | 0.166 / 0.138 / 0.182 | 0.166 / 0.117 / 0.197 | 0.088 / 0.096 / 0.140 | 0.118 / 0.103 / 0.162 |

## Target: per-lead skill (`V2-F-M`, 60 val random windows, ss vs min(persistence, climatology))

| Model | day 1 | day 2 | day 4 | day 6 | day 8 | day 10 |
|---|---|---|---|---|---|---|
| r3_cnn | +0.210 | +0.155 | +0.035 | -0.022 | -0.016 | -0.035 |
| F0 | +0.207 | +0.151 | +0.029 | -0.023 | -0.018 | -0.035 |
| F4 | +0.207 | +0.151 | +0.029 | -0.022 | -0.017 | -0.034 |
| F5 | +0.208 | +0.153 | +0.034 | -0.021 | **-0.010** | **-0.026** |

Paired differences (mean +- 2 standard errors):

| Pair | day 1 | day 2 | day 4 | day 6 | day 8 | day 10 |
|---|---|---|---|---|---|---|
| F4 - F0 | -0.0000+-0.0003 | +0.0001+-0.0005 | -0.0000+-0.0010 | +0.0008+-0.0016 | +0.0007+-0.0014 | +0.0014+-0.0018 |
| F5 - F0 | +0.0004+-0.0014 | +0.0016+-0.0026 | +0.0047+-0.0051 | +0.0019+-0.0057 | +0.0076+-0.0066 | +0.0091+-0.0074 |
| F4 - r3_cnn | -0.0028+-0.0014 | -0.0040+-0.0025 | -0.0060+-0.0041 | -0.0009+-0.0049 | -0.0005+-0.0050 | +0.0015+-0.0051 |
| F5 - r3_cnn | -0.0024+-0.0016 | -0.0025+-0.0030 | -0.0013+-0.0042 | +0.0002+-0.0060 | +0.0064+-0.0068 | +0.0093+-0.0090 |

## Day-8 variance ratios (random windows; t2m, tp, u10, v10, msl, tcwv; 1 = sharp)

| Model | val | test |
|---|---|---|
| r3_cnn | 0.98 / 1.08 / 0.77 / 0.72 / 0.93 / 0.93 | 1.00 / 0.96 / 0.86 / 0.79 / 0.88 / 0.95 |
| F0 | 0.98 / 1.14 / 0.77 / 0.73 / 0.93 / 0.93 | 1.00 / 0.96 / 0.87 / 0.80 / 0.89 / 0.94 |
| F4 | 0.98 / 1.14 / 0.77 / 0.74 / 0.92 / 0.92 | 0.99 / 0.95 / 0.87 / 0.80 / 0.88 / 0.94 |
| F5 | 0.99 / 0.96 / 0.74 / 0.75 / 0.94 / 0.89 | 1.00 / 0.78 / 0.83 / 0.80 / 0.89 / 0.91 |

F5 is blurrier in rain (test 0.96 -> 0.78) and slightly in u10/tcwv. Blur points stay 10/10 (9.59 on val events).

## v2 checks and guards

| Model | Fani err d1/d2/d3 km | Fani landfall lat | Wind-shock W-coast rain (share neg) | Edge / interior RMSE | Reach day-1 rmax val/test | 10-day rollout (2 thr) | test_engine / null |
|---|---|---|---|---|---|---|---|
| r3_cnn | 350 / 245 / 809 | +2.0 | -1.29 mm (92%) | 0.441 / 0.503 | 9.90 / 9.67 | 0.27-0.36 s | pass / 0.0 |
| F0 | 350 / 245 / 809 | +2.0 | -1.24 (83%) | 0.444 / 0.504 | 9.90 / 9.67 | - | - |
| F4 | 350 / 245 / 809 | +2.0 | -1.22 (83%) | 0.444 / 0.504 | 9.90 / 9.67 | 0.10 s | pass / 0.0 |
| F5 | 350 / 245 / 809 | +2.0 | -1.26 (92%) | 0.444 / 0.502 | 9.90 / 9.67 | 0.06 s | pass / 0.0 |

Repo scan 15/15. Rollout times were taken on a shared CPU, so compare them only side by side. Amphan (test year, report only):
d2 786 km and landfall lat -7.0 for all three, the same as r3_cnn.

## Verdicts

- **F4 (unroll noise 0.1): loses.** Target: it is the same as F0 at every lead (paired difference <= 0.0014, inside 2 SE),
  and the day-8 ratios are the same. `V2-WIN` vs r3_cnn: val mean -0.12 (ok), but test mean -0.56 (> 0.2) and test shocks
  12.75 < 13.12. Guards hold. At this size the noise changes nothing: the net removes white noise easily, and its own
  errors are smooth and large-scale, not white.
- **F5 (10-day unroll): loses.** Target: the only positive signal of the round. At days 8 and 10 it is the best model
  (+0.0076 and +0.0091 vs F0, +0.0064 / +0.0093 vs r3_cnn), and at day 4 +0.0047 vs F0. But every gain is under 2 paired SE,
  so it is not beyond noise under `V2-F-M`. `V2-WIN`: val mean -0.73 (val events 65.87: CSI 0.088 and blur 9.59) and
  test mean -0.46; test shocks 12.75. It also blurs rain at day 8 (test ratio 0.78). Guards hold.
- F0 itself (more training, no change) loses to r3_cnn on test (-0.95) and on per-lead skill at days 1-4 (+0.003 to +0.006
  for r3_cnn, each beyond 2 SE). So "fine-tune 4 more epochs at lr 3e-4" slightly hurts, and both F runs inherit that.
- Not combined with other F winners (none). If the team retries, the evidence-based next steps are:
  - F5 with 3 seeds, to see whether the late-lead gain is real;
  - F4 with spatially smooth noise, or a much larger sigma (~0.3-0.5, close to day-1 error).
  Each needs its own spec entry first.

## Errors found, classified

| Type | Where | What |
|---|---|---|
| Spec error (missing detail) | V2-F4 / V2-F5 | No flag name, noise size, rain handling, F5 selection rule or numeric "beats F0" rule. Fixed in this branch's spec first (`--unroll-noise`, `V2-F-M` with 2 paired SE) |
| Spec error (rationale, fixed before training) | V2-F4 | The first text said 3x3 convolutions average white noise down; the residual step carries it 1:1. Corrected after the smoke test |
| Spec error (hypothesis) | V2-F4 | White i.i.d. noise at 0.1 std does not resemble the model's own (smooth, large-scale) errors, so it teaches nothing (F4 = F0 within 0.0014) |
| Spec error (protocol) | V2-F5 | Choosing by the 10-day val loss took epoch 3. Its loss differs from F0's 8-day loss, so the "best epoch" rules differ between the two runs |
| Code error (tooling, not repo code) | scoring | A scoring script in the shared scratchpad had the same name as another agent's, was overwritten, and re-scored `v2_f0`/`v2_f1` in `wt-f-sample` (duplicate rows there: leaderboard 03:41:53, 03:42:06, 03:42:52, 03:43:04; v2_checks 03:42:13, 03:43:10; heavy_bias 03:42:22-03:43:22). Stopped, not edited, since another agent owns that worktree. The rows are deterministic duplicates. All scoring for this branch was redone with scripts kept in `wt-f-growth/runs/fgrowth/` |
| Code (feature added) | `train_nn`, `twin/nets/rollout.py` | `--unroll-noise` (default 0, training only); `training/lead_skill.py` (per-lead curve with paired SE) |
