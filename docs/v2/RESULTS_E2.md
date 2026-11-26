# V2-E2 results: soft-CSI loss on rain (branch `v2/e2-softcsi`)

**Verdict: does not win (V2-WIN fails).** The target metric (heavy-rain `csi_skill`) went down on 3 of 4
window sets, both auto means fell by more than 0.2 points, and test shocks fell from 13.12 to 12.75.
No extra seeds.

**E2b (soft CSI alone, no bias guard) also does not win** (see the E2b section at the end). Val looks slightly
better (+0.16 mean, csi_skill +0.002/+0.005, within noise), but the test mean is -0.88 and test shocks drop to 12.75.
Raw heavy-rain CSI is unchanged against the baseline; only the over-forecast grows. E2 is closed with both
allowed settings tried.

## Settings

- Recipe = `r3_cnn` baseline (CNN default width, `--rain-head two_part`, sqrt rain, heavy weight 3, spread 0.1,
  spectral 0.05, curriculum 1:4,2:3,4:3,8:2, batch 32, lr 1e-3, seed 0) + **one change**:
  `--csi-loss 0.5 --csi-tau 2 --bias-guard 0.1` (spec `V2-E2`, concrete form refined in this branch).
- Loss term per batch: rain mm/day decoded from the anomaly as in `PhysScale.rain_mm` (prediction in the forward
  pass is the hard-gated rain, with straight-through gradient); `s = sigmoid((rain_mm - 10)/2)`; soft counts summed
  over samples, unrolled steps and cells; `0.5*(1 - H/(H+M+F+1)) + 0.1*|log((sum s_p + 1)/(sum s_t + 1))|`.
  Checkpoint selection unchanged (plain-MSE val at unroll 8). Nothing is added to the inference fields of the checkpoint.
- One weight setting tried (limit 2). Smoke test (width 8, 2 short epochs) ran clean. On `r3_cnn` the term was about
  0.30 (soft CSI) + 0.01 (bias) next to a base loss of about 0.7.
- Training: epochs 1-8 on 1 thread. After the v1.1 retrain finished, the run was stopped at a saved epoch boundary
  and resumed (`--resume`) with 2 threads for epochs 9-12, on the coordinator's instruction (`V2-RES` allows this
  once the retrain is done). Best val MSE 0.7941 (epoch 12). Log: `runs/logs/v2_e2.log`.

## Numbers (baseline `r3_cnn` from spec `V2-P` / `runs/leaderboard.csv`)

Judge auto points (of 70), window sets events / random / named:

| | val | val mean | test | test mean |
|---|---|---|---|---|
| r3_cnn | 67.51 / 68.12 / 68.12 | 67.92 | 68.12 / 67.65 / 68.12 | 67.96 |
| **v2_e2** | 66.96 / 66.56 / 68.12 | **67.21 (-0.71)** | 67.13 / 66.70 / 67.75 | **67.19 (-0.77)** |

Uncapped:

| metric | r3_cnn val | v2_e2 val | r3_cnn test | v2_e2 test |
|---|---|---|---|---|
| ss events | 0.168 | 0.164 | 0.167 | 0.168 |
| ss random | 0.138 | 0.137 | 0.118 | 0.117 |
| ss named | 0.185 | 0.182 | 0.194 | 0.208 |
| **csi_skill events** | 0.098 | **0.095** | 0.108 | **0.096** |
| **csi_skill random** | 0.105 | **0.090** | 0.105 | **0.098** |
| csi_skill named | 0.140 | 0.158 | 0.162 | 0.161 |
| shocks (causal pts /15) | 13.12 | 13.12 | 13.12 | **12.75** |
| reach rmax day 1 (cells) | 9.9 | 9.67 | 9.67 | 9.67 |
| null test (5 event windows) | ok | ok (5/5) | ok | ok (5/5) |

Test shocks: the warm-ocean shock on the test cyclone window exceeded 5 std ("stable" failed). monsoon_stall
"propagates" fails on every window, for the baseline too.

Heavy-rain diagnostic (60 random windows, CSI at > 10 mm/day and frequency bias = predicted / true heavy cells):

| | d1 CSI / bias | d4 CSI / bias | d8 CSI / bias |
|---|---|---|---|
| r3_cnn val | 0.552 / 1.09 | 0.422 / 1.17 | 0.361 / 1.36 |
| v2_e2 val | 0.555 / 1.02 | 0.404 / 0.99 | 0.359 / 1.11 |
| r3_cnn test | 0.535 / 1.11 | 0.398 / 1.35 | 0.299 / 1.52 |
| v2_e2 test | 0.531 / 1.02 | 0.396 / 1.14 | 0.294 / 1.24 |

`training.v2_checks`:

| | r3_cnn | v2_e2 |
|---|---|---|
| Fani centre error d1/d2/d3 (km) | 350 / 245 / 809 | 350 / 755 / 914 |
| Fani landfall-lat error | +2.0 | +5.0 |
| wind-shock west-coast rain (mm), share < 0 | -1.29, 92% | -0.81, 83% |
| edge / interior day-4 RMSE (std) | 0.441 / 0.503 | 0.446 / 0.506 |
| ms per rollout (v2_checks) | 86 | 148 |
| Amphan d2 / landfall-lat (TEST ONLY, not used) | 786 / -7.0 | 862 / -4.0 |

The ms difference is not a model difference: same network, same size. Both v2_e2 scoring runs used
`TWIN_THREADS=1` on a machine shared with other experiments. Measured on 2 threads: median 90 ms, max 104 ms per
10-day rollout.

## V2-WIN check

| condition | result |
|---|---|
| target `csi_skill` val+test improves beyond noise | **no**: events val 0.098 -> 0.095, test 0.108 -> 0.096; random val 0.105 -> 0.090, test 0.105 -> 0.098 |
| val auto mean not lower than baseline by > 0.2 | **no**: -0.71 |
| test auto mean not lower than baseline by > 0.2 | **no**: -0.77 |
| shocks not lower | **no** on test (13.12 -> 12.75); equal on val |
| `tests/test_engine.py` passes | yes (run on v2_e2 through a wrapper; also passes on the active model) |
| null test exact | yes (5/5 windows, both splits) |
| reach day-1 max <= 11 | yes (9.67) |
| 10-day rollout < 1 s on 2 threads | yes (90 ms) |
| repo scan 15/15 | yes |

Why it failed: the bias guard worked. It pulled the heavy-rain frequency bias from 1.09-1.52 down to 1.0-1.24.
But with an uncertain forecast, CSI is highest at a bias above 1, so trimming the over-forecast removed hits
as well as false alarms. The soft-CSI term did not make up the difference. The model also became less responsive
to the wind shock (-0.81 vs -1.29 mm) and lost Fani track accuracy on days 2-3.

Follow-up: the second allowed setting, `--csi-loss 0.5 --bias-guard 0`, was run as E2b (below).

## Errors found

- **Spec error (missing detail), fixed:** `V2-E2` defined no flags, tau, weights, eps or aggregation of the
  soft counts. Made concrete in `docs/SPEC.md` before coding (change-log line added).
- **Spec error (design):** the `V2-E2` bias guard aims for frequency bias = 1, which conflicts with the CSI target.
  The CSI-best bias for a blurred forecast is above 1, and the baseline sits at 1.1-1.5. Measured here: bias moved
  toward 1 and CSI dropped. A guard should only penalise bias outside a band (e.g. < 0.8 or > 1.6), or be dropped.
- **Spec error (missing condition):** `V2-P` asks for "ms" without fixing a thread count or machine load.
  `ms_per_rollout` in `leaderboard.csv` and `v2_checks.csv` depends on `TWIN_THREADS` and on other jobs running,
  so it cannot be compared across experiments. The comparable number is the `V2-WIN` 2-thread timing.
- **Code note (latent, not a spec mismatch):** `train_nn --resume` checks only `--curriculum`/`--batch` against
  the saved state. A resume with different loss flags would silently mix recipes. It did not happen here: the
  resumed command was identical apart from `--threads`.
- No code error found against `V2-E2`. With the flags off (default 0), the training path is the same as before
  (no extra term, `PhysScale` built only when it is needed).

---

## E2b: soft CSI alone (`--csi-loss 0.5 --csi-tau 2 --bias-guard 0`)

This is the second and last allowed setting, asked for by the coordinator after the E2 finding (spec `V2-E2` E2b line).
The recipe is otherwise identical to E2 and `r3_cnn`. It trained on 3 threads in one run, with no restart. Best val
MSE was 0.8061 at epoch 11; E2 reached 0.7941. Logs: `runs/logs/v2_e2b.log`, `runs/logs/v2_e2b_score.log`.

Judge auto points (of 70), window sets events / random / named:

| | val | val mean | test | test mean |
|---|---|---|---|---|
| r3_cnn | 67.51 / 68.12 / 68.12 | 67.92 | 68.12 / 67.65 / 68.12 | 67.96 |
| v2_e2 | 66.96 / 66.56 / 68.12 | 67.21 (-0.71) | 67.13 / 66.70 / 67.75 | 67.19 (-0.77) |
| **v2_e2b** | 68.00 / 68.12 / 68.12 | **68.08 (+0.16)** | 67.75 / 65.73 / 67.75 | **67.08 (-0.88)** |

Uncapped:

| metric | r3_cnn val | v2_e2b val | r3_cnn test | v2_e2b test |
|---|---|---|---|---|
| ss events | 0.168 | 0.164 | 0.167 | 0.176 |
| ss random | 0.138 | 0.138 | 0.118 | **0.112** (28.0/30 points) |
| ss named | 0.185 | 0.184 | 0.194 | 0.196 |
| **csi_skill events** | 0.098 | **0.100** | 0.108 | **0.105** |
| **csi_skill random** | 0.105 | **0.110** | 0.105 | **0.101** |
| csi_skill named | 0.140 | 0.132 | 0.162 | 0.169 |
| shocks (points of 15) | 13.12 | 13.12 | 13.12 | **12.75** |
| reach rmax day 1 (cells) | 9.9 | 9.9 | 9.67 | 9.9 |
| null test (5 event windows) | ok | ok (5/5) | ok | ok (5/5) |

Test shocks: the same failure as E2. The warm-ocean shock on the test cyclone window exceeded 5 std.

Heavy-rain diagnostic (60 random windows, CSI > 10 mm/day and frequency bias):

| | d1 CSI / bias | d4 CSI / bias | d8 CSI / bias |
|---|---|---|---|
| r3_cnn val | 0.552 / 1.09 | 0.422 / 1.17 | 0.361 / 1.36 |
| v2_e2b val | 0.556 / 1.16 | 0.424 / 1.28 | 0.360 / 1.49 |
| r3_cnn test | 0.535 / 1.11 | 0.398 / 1.35 | 0.299 / 1.52 |
| v2_e2b test | 0.537 / 1.20 | 0.392 / 1.50 | 0.301 / 1.65 |

`training.v2_checks`:

| | r3_cnn | v2_e2 | v2_e2b |
|---|---|---|---|
| Fani centre error d1/d2/d3 (km) | 350 / 245 / 809 | 350 / 755 / 914 | 350 / 739 / 1012 |
| Fani landfall-lat error | +2.0 | +5.0 | -6.0 |
| wind-shock west-coast rain (mm), share < 0 | -1.29, 92% | -0.81, 83% | -1.10, 75% |
| edge / interior day-4 RMSE (std) | 0.441 / 0.503 | 0.446 / 0.506 | 0.454 / 0.506 |
| ms per rollout (v2_checks, 1 thread) | 86 | 148 | 75 |
| Amphan d2 / landfall-lat (TEST ONLY, not used) | 786 / -7.0 | 862 / -4.0 | 676 / -6.0 |

### V2-WIN check (E2b)

| condition | result |
|---|---|
| target `csi_skill` val+test improves beyond noise | **no**: val +0.002 / +0.005 (noise level), test -0.003 / -0.004 |
| val auto mean not lower than baseline by > 0.2 | yes (+0.16) |
| test auto mean not lower than baseline by > 0.2 | **no**: -0.88 (test random SS 0.118 -> 0.112) |
| shocks not lower | **no** on test (13.12 -> 12.75); equal on val |
| `tests/test_engine.py` passes | yes (run on v2_e2b through a wrapper) |
| null test exact | yes (5/5 windows, both splits) |
| reach day-1 max <= 11 | yes (9.9) |
| 10-day rollout < 1 s on 2 threads | yes (median 48 ms, max 59 ms) |
| repo scan 15/15 | yes |

**Verdict: E2b does not win.** Without the guard, the soft-CSI term mostly made the model forecast heavy rain more
often. Bias went up at every lead (to 1.16-1.65), while raw CSI stayed within ±0.006 of the baseline. Its points
moved with that: val events and random reach the CSI cap (15/15), but test does not improve. It also cost test-year
mean skill (random SS 0.112) and Fani track accuracy. The judge's capped CSI points (full at 0.10) leave little
to gain over a baseline already at 0.098-0.108. Conclusion for E2: a soft-CSI loss at this weight is not a lever for
this model, with or without the bias guard. Both allowed settings are used, so E2 is closed.

Errors found in E2b: none new. The shared scratchpad was overwritten by other sessions again, which is a process
note rather than a code or spec error; E2 used uniquely named helper scripts.
