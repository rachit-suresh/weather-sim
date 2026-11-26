# V2-E2 results: soft-CSI loss on rain (branch `v2/e2-softcsi`)

**Verdict: does not win (V2-WIN fails).** The target metric (heavy-rain `csi_skill`) went down on 3 of 4
window sets, both auto means fell by more than 0.2 points, and test shocks fell from 13.12 to 12.75.
No extra seeds.

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

Possible next step (not run, to save time): the second allowed setting, `--csi-loss 0.5 --bias-guard 0`, tests
the soft-CSI term alone.

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
