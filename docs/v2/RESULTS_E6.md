# v2 experiment E6: learned edges (zero padding + random crops)

Branch `v2/e6-edges`, model `artifacts/models/v2_e6.pt`, spec `V2-E6` (section 14, refined in this branch).

## What changed

Baseline recipe `r3_cnn` (spec `V2-P`) plus two changes, run together in ONE training. The spec says one change per
experiment; `V2-E6` was updated first (spec change log) because there was time for only one run:

- (a) `pad="zeros"` CNN option (`--net 'pad="zeros"'`, stored in the checkpoint `net_config`; old checkpoints and the
  default stay `replicate`). In anomaly space a zero neighbour means "normal weather outside the box".
- (b) `--random-crop`: each training batch is cut to one random sub-window (height 24-32, width 28-36, random
  position) for inputs, targets, masks, static maps and latitude weights. The edge-distance map is recomputed for
  the crop. Validation and inference use the full 32 x 36 grid.

Command: `python -m training.train_nn --family cnn --name v2_e6 --rain-head two_part --threads 1 --no-eval --resume
--net 'pad="zeros"' --random-crop`. Epochs 1-9 ran on 1 thread. After the v1.1 retrain finished, the coordinator
asked for a restart, so epochs 10-12 ran on 2 threads (`--resume` from the epoch-9 state, RNG state restored).
Best val loss 0.8145.

## Numbers vs baseline `r3_cnn`

Judge points (auto, of 70), `training.evaluate`, `TWIN_THREADS=1`:

| Set | r3_cnn val | v2_e6 val | r3_cnn test | v2_e6 test |
|---|---|---|---|---|
| events | 67.51 | 67.98 | 68.12 | 67.38 |
| random | 68.12 | 67.67 | 67.65 | **61.88** |
| named | 68.12 | 68.12 | 68.12 | 67.38 |
| **mean** | 67.92 | 67.92 (+0.00) | 67.96 | **65.55 (-2.41)** |

Uncapped:

| Metric | r3_cnn val | v2_e6 val | r3_cnn test | v2_e6 test |
|---|---|---|---|---|
| ss events / random | 0.168 / 0.138 | 0.160 / 0.129 | 0.167 / 0.118 | 0.179 / 0.106 |
| ss named | 0.185 | 0.180 | 0.194 | 0.203 |
| csi_skill events / random | 0.098 / 0.105 | 0.110 / 0.097 | 0.108 / 0.105 | 0.126 / 0.086 |
| csi_skill named | 0.140 | 0.153 | 0.162 | 0.190 |
| shocks (of 15) | 13.12 | 13.12 | 13.12 | **12.38** |
| reach r95 / rmax day 1 (cells) | 2.08 / 9.9 | 4.13 / 9.22 | 1.88 / 9.67 | 1.88 / 9.22 |

Shock checks that fail (5 event windows per split):

| Split | r3_cnn | v2_e6 |
|---|---|---|
| val | monsoon_stall propagates: 5/5 windows | monsoon_stall propagates: 5/5 windows (unchanged) |
| test | monsoon_stall propagates 4/5; warm_ocean stable 1/5 | monsoon_stall propagates 4/5; warm_ocean stable **2/5**; warm_ocean propagates **1/5** |

Null test exact in every window (both models).

`training.v2_checks`:

| Check | r3_cnn | v2_e6 |
|---|---|---|
| **edge-band day-4 RMSE (std)** (target) | 0.441 | **0.445** (worse) |
| interior day-4 RMSE (std) | 0.503 | 0.509 |
| edge / interior | 0.876 | 0.876 |
| Fani centre error d1/d2/d3 (km) | 350 / 245 / 809 | 247 / 430 / 884 |
| Fani landfall-lat error (deg) | +2.0 | +1.0 |
| wind shock: west-coast rain change, share < 0 | -1.29 mm, 92% | -1.39 mm, 92% |
| Amphan d2 / landfall lat (TEST ONLY, not used) | 786 km / -7.0 | 104 km / 0.0 |
| ms per 10-day rollout | 86 (logged earlier) | 146 (logged) |

The ms figures were logged under different machine load. Timed side by side under the same load (1 thread, 12
other trainings running), r3_cnn takes 159 ms and v2_e6 takes 146-149 ms, so speed is unchanged. On 2 threads
v2_e6 takes 134 ms (< 1 s guard).

## Verdict under `V2-WIN`: does not win

- Target: edge-band RMSE did not improve (0.441 -> 0.445). The edge/interior ratio is unchanged and the interior
  got slightly worse too. monsoon_stall "propagates" still fails in 5/5 val windows and 4/5 test windows.
- Test auto mean falls 2.41 points, far past the 0.2 limit. Most of the drop is random-window skill (ss 0.118 ->
  0.106, csi_skill 0.105 -> 0.086). Val random ss also falls (0.138 -> 0.129).
- Shocks on test are lower (13.12 -> 12.38: warm_ocean unstable in one more window).
- Guards hold: `tests/test_engine.py` contract passes on v2_e6 (run with the engine pointed at v2_e6;
  `active.json` untouched); null test exact; reach day-1 max 9.22 <= 11 cells; 10-day rollout 134 ms on 2 threads;
  repo scan 15/15.
- Gains seen (val csi_skill events 0.110, named sets, Fani landfall latitude) are within single-seed noise and do not
  offset the test drop. Per `V2-WIN`, no extra seeds.

Likely reasons (not tested separately, since (a) and (b) were combined):

1. Zero padding only means "normal weather" for the 18 anomaly channels. The other 11 input channels (target-day
   climatology, land/sea mask, orography, sin/cos, edge distance) are not anomalies, so a zero there is an arbitrary
   value at the first layer. Hidden layers have no "normal" value either.
2. A random crop hides real, known weather just outside the crop and asks the net to forecast as if it were unknown.
   That adds target noise to cells that are interior at inference. It does not teach the net real inflow, because
   the dataset has nothing outside 32 x 36 (`CON-6`).

If edges are revisited, try (a) alone first; inflow is better handled by the E7 semi-Lagrangian design.

## Errors found

- **Spec error (fixed):** `V2-E6` was underspecified (crop sizes, edge map per crop, where the pad option is stored)
  and clashed with `V2-P` "one change per experiment". It now has a concrete text and a change-log line.
- **Spec error (rationale, not fixed):** `V2-E6` (and the methods report) say zero padding equals "climatology
  outside". That holds only for the anomaly input channels, not for the climatology, static and time channels or
  hidden features (reason 1 above).
- **Spec gap (not fixed):** `V2-WIN` says "beyond noise" but gives no noise level for the v2_checks metrics (single
  seed, 30 windows). It did not matter here (the change is in the wrong direction).
- **Code error (latent, fixed):** `make_input`, `edge_cells`/`edge_distance`/`edge_relax_map` and the `SHP-8`
  shock box were hard-wired to 32 x 36. Any sub-window input would have crashed or been mislabelled. They now take
  the size from the tensor; full-grid behaviour is unchanged (old checkpoints load and give the same output).
- **Process note:** the 1 -> 2 thread restart after epoch 9 means the run is not bit-identical to a pure 1-thread
  run (float summation order). The data order and RNG were restored by `--resume`.
