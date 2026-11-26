# v2 experiment E5: likelihood-trained rain amount (spec `V2-E5`)

Branch `v2/e5-rain-likelihood`, model `artifacts/models/v2_e5.pt`, tags `v2_e5` (val) / `test_v2_e5` (test).

**Verdict: LOSES (`V2-WIN`).** The target metrics (heavy-rain `csi_skill` and rain RMSE) got worse on validation
AND test years. The auto means fell by more than 0.2 (val -2.37, test -0.67). Shock points went up (+1.13 val,
+1.50 test), and every guard holds. The experiment does not get extra seeds.

## What changed (one thing)

Baseline `r3_cnn` recipe (CNN, `RAIN-3` two-part head, sqrt rain, spread + spectral losses, curriculum
1:4,2:3,4:3,8:2, train 2005-2017) + `--rain-loss gamma`:

```
python -m training.train_nn --family cnn --name v2_e5 --rain-head two_part --rain-loss gamma --threads 1 --no-eval --resume
```

- The gate is unchanged: hard gate with a straight-through gradient, BCE 0.5 on rain > 1 mm. It is the Bernoulli part.
- On the rain channel, the squared error (and its 3x heavy-rain weight) is replaced by a gamma negative
  log-likelihood. It applies on wet cells only (true rain > 1 mm). The mean is `mu = max(amount^2, 0.1)` mm/day,
  where the amount is the ungated sqrt-space output. It uses one learned global shape `k`.
- The other 5 channels, the spread and spectral terms, and the validation loss used to pick the best epoch are
  unchanged.
- The served forecast is the gamma mean (`amount^2` where the gate is open). That is the same decode as `RAIN-3`:
  deterministic, with no new output channel. The checkpoint stores `rollout.rain_amount = {loss: gamma, shape: 1.887}`
  for information only.
- Training was stable (no NaN). `k` went from 1 to 2.8 in the 1-step stage and settled at 1.88-1.96 in the unrolled
  stages. The Tweedie fallback was therefore not needed and was not run (it is implemented: `--rain-loss tweedie`).
- Best validation loss 0.8122 (epoch 11 of 12).
- Threads: 1 thread for epochs 1-8. After the v1.1 retrain finished, the coordinator asked for a restart with
  `--threads 2 --resume` from the epoch-8 state for epochs 9-12. Only floating-point summation order differs.

## Judge points (auto /70 per window set)

| | val events | val random | val named | **val mean** | test events | test random | test named | **test mean** | shocks val / test |
|---|---|---|---|---|---|---|---|---|---|
| `r3_cnn` (baseline) | 67.51 | 68.12 | 68.12 | **67.92** | 68.12 | 67.65 | 68.12 | **67.96** | 13.12 / 13.12 |
| `v2_e5` | 61.59 | 65.96 | 69.11 | **65.55** | 64.60 | 67.65 | 69.62 | **67.29** | **14.25 / 14.62** |
| difference | -5.92 | -2.16 | +0.99 | **-2.37** | -3.52 | 0.00 | +1.50 | **-0.67** | +1.13 / +1.50 |

The losses are almost all heavy-rain CSI points (val events 7.5/15, val random 11.7/15, test events 10.0/15). The
gains on the named sets come from the higher shock points (they are added to every set).

## Uncapped numbers

| | ss events | ss random | ss named | csi_skill events | csi_skill random | csi_skill named |
|---|---|---|---|---|---|---|
| val `r3_cnn` | 0.168 | 0.138 | 0.185 | 0.098 | 0.105 | 0.140 |
| val `v2_e5` | 0.157 | 0.134 | 0.176 | **0.050** | **0.078** | **0.099** |
| test `r3_cnn` | 0.167 | 0.118 | 0.194 | 0.108 | 0.105 | 0.162 |
| test `v2_e5` | 0.160 | 0.116 | 0.194 | **0.067** | **0.093** | **0.146** |

Rain diagnostics (mm/day, latitude-weighted, at leads 1/4/8 on the same 60 random windows that `training.evaluate`
uses, plus the 5 event windows; computed with a throwaway script that calls `twin.engine.Engine`):

| | rain RMSE d1 / d4 / d8 (mean) random | rain RMSE mean, events | heavy (>10 mm) count / true count | raw CSI at 10 mm | mean bias |
|---|---|---|---|---|---|
| val `r3_cnn` | 3.70 / 5.38 / 6.08 (5.05) | 8.42 | 1.21 | 0.436 | -0.16 |
| val `v2_e5` | 3.90 / 5.47 / 6.09 (5.15) | 8.67 | 1.10 | 0.420 | -0.14 |
| test `r3_cnn` | 3.97 / 5.89 / 6.50 (5.45) | 7.30 | 1.32 | 0.400 | +0.02 |
| test `v2_e5` | 4.09 / 5.91 / 6.51 (5.50) | 7.65 | 1.21 | 0.391 | -0.01 |

Rain RMSE is 1-2% worse on random windows (mostly on day 1) and 3-5% worse on event windows. Heavy-rain
over-forecasting fell (frequency bias moved closer to 1), but the hit rate fell more, so CSI dropped.

## v2_checks (`runs/v2_checks.csv`)

| | Fani err km d1 / d2 / d3 | Fani landfall lat err | wind-shock west-coast rain | edge / interior std | ms |
|---|---|---|---|---|---|
| `r3_cnn` | 350 / 245 / 809 | +2.0 | -1.29 mm (92% neg) | 0.441 / 0.503 | 86 |
| `v2_e5` | 247 / 755 / 914 | +5.0 | -1.09 mm (92% neg) | 0.445 / 0.506 | 119 |

Amphan (test year, reported only, never used to decide): d2 error 237 km (baseline 786), landfall latitude error -1.0
(baseline -7.0). The architecture is identical to the baseline, so the ms difference is machine load: other
experiments were training during scoring. With 2 threads on a quiet moment, a 10-day rollout takes 86 ms median.

## Guards (`V2-WIN`)

- `tests/test_engine.py`: passes (active model). The same contract test passes with `v2_e5` swapped in: shape,
  validity (rain >= 0, finite, t2m range), determinism, zero perturbation == null, and a local shock response.
  pytest is not installed in the venv, so it ran as `python tests/test_engine.py`.
- Null test exact: `null_ok` on all 5 event windows, val and test.
- Reach probe: day-1 rmax 9.67 (val) / 9.45 (test) cells, within 11.
- 10-day rollout under 1 s on 2 threads: 86 ms median, 93 ms max.
- Repo scan 15/15.

## Why it lost

In sqrt space, the gamma likelihood's gradient for the amount `a` is about `4k (a - sqrt(y)) / a^2`. The baseline's
sqrt-space squared error gives `2 (a - sqrt(y))`, times 3 on heavy cells. So, relative to the baseline, the
likelihood weights each cell's error by about `1/mu`. A 16 mm/day cell gets about 1/16 the weight of a 1 mm/day
cell, and it also loses the 3x heavy weight. Gamma variance grows as `mu^2`, so the likelihood treats big errors on
heavy rain as expected noise. That is the opposite of what the methods report hoped for ("keeps the tail"). The
mean-unbiased estimator shows in the data (bias near 0, frequency bias nearer 1), but at 10 mm the hits drop. Tweedie
(power 1.5) would weight errors by about `1/sqrt(mu)`. That is milder, but it still down-weights heavy cells relative
to the baseline, so it is not expected to win either.

The higher shock points (+1.1 / +1.5) are the one positive. On val, monsoon_stall "propagates" failed on 2 of 5
windows (warm_ocean passed all). On test, warm_ocean "stable" failed on 1 window. This run does not show why. The judge sees only 5 windows, so this is within noise for one seed, and it is
not the target.

## Errors found

1. **Spec error (gap), fixed before coding.** `V2-E5` did not say where the likelihood applies, what happens to the
   `RAIN-2` heavy weight, how the shape is parameterised, which value is served, or how rain RMSE is measured. The
   spec now covers all of these (change-log line "v2/e5").
2. **Spec error (hypothesis), found by the result.** `V2-E5` (from the methods report) expected the likelihood to
   improve heavy rain. For gamma (and Tweedie), the per-cell weight falls as `1/mu^(p-1)`, so heavy cells count less
   than under the baseline's 3x weighted squared error. The spec now records the result.
3. **No code errors found in the experiment path.** Smoke checks passed: train, resume, Tweedie path, and the default
   path unchanged (`--rain-loss mse` gives the same loss and optimiser setup as before). The engine checks passed:
   zero perturbation == none, rain >= 0, finite. A `float(tensor with grad)` warning in the new epoch print was fixed
   during the smoke test.
4. **Process note.** The `V2-RES` 1-thread rule was relaxed to 2 threads for epochs 9-12, on the coordinator's
   instruction, after the v1.1 retrain finished. The ms numbers in `V2-P` depend on machine load, so they are only
   comparable when measured on a quiet machine.
