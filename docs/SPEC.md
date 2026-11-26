# weather-sim specification

**Version:** 1.1 (2026-11-26). **Status:** active.
**Rule:** this file is the source of truth. Every change is made here FIRST (with a change-log line), then in
code. Each requirement has an ID (e.g. `REP-2`) so commits, issues and reviews can point at it.

**Error classes.** Every problem found is labelled one of:
- **Code error:** the code does not do what this spec says. Fix the code.
- **Spec error:** the spec is wrong, missing something, or contradicts the contest rules or the data. Fix the
  spec first (new version + change-log line), then the code.

Background docs (explanations, not requirements): `docs/HOW_THE_MODELS_WORK.md`, `docs/MODEL_FAMILIES.md`,
`docs/GRADING_NOTES.md`, `docs/RESULTS_SWEEP_1.md`, `docs/PHYSICS_HYBRID.md`.

---

## 1. Contest contract (fixed by the organisers)

- `CON-1` `python serve.py --port 8000` at the repo root starts the server on `0.0.0.0`, offline, 4 CPU, 6 GB RAM,
  no GPU, read-only repo (write only to /tmp). `/health` must report ready within 120 s.
- `CON-2` `POST /simulate` follows `API_CONTRACT.md` exactly: input `[3,6,32,36]` physical units oldest first,
  `doy_sincos` of the last input day, `steps`, `perturbation` `[6,32,36]` delta or null; output
  `[steps,6,32,36]` physical units. Under 5 s per 10-step call including JSON.
- `CON-3` Output validity: finite, rain >= 0, t2m in [233.15, 333.15] K.
- `CON-4` Zero perturbation output == null perturbation output (max |diff| <= 1e-3 channel std).
- `CON-5` Repo scan 15/15 (`scripts/check_repo.py`): no simulate route string in non-server `.py` files, no dates
  or weather-data-source names in server code, no climate data files or multi-year arrays, weights in the repo,
  training code present, README at root, frontend present.
- `CON-6` No training data from 2023 or later (hidden test period). No outside weather data at inference.

## 2. Architecture

- `ARC-1` `serve.py` (HTTP only) -> `twin/engine.py` (shock, clamps, fallback) -> active model from
  `artifacts/active.json`. Server-side code lives only in `serve.py` and `twin/`.
- `ARC-2` Training, fitting and evaluation live in `training/` and are never imported by the server.
- `ARC-3` Crash fallback: if the active model raises or returns non-finite values, the damped-anomaly baseline
  answers (deterministic).
- `ARC-4` Dashboard in `dashboard/` (HTML/CSS/JS), talks to the server over HTTP only; contract and DO NOTs in
  `docs/DASHBOARD_API.md`. Owned by the visualisation teammate.

## 3. Data and validation (decision H)

- `DAT-1` Training data: `india_climate_train.nc` 2005-2021, kept outside the repo.
- `DAT-2` Split by year: train 2005-2017, validate 2018-2019, test 2020-2021. Test is used only for final checks.
- `DAT-3` Climatology: 2005-2021, 15-day smoothed, 366 slots (kit builder), shipped in `artifacts/climatology.npz`.
- `VAL-1` Local grader = organisers' scoring code on: 5 auto-picked event windows, 60 random windows,
  named real events (`VAL-2`), the judge's 3 shocks on every event window, and the reach probe.
  Each window set gets an "auto /70" score = RMSE skill points (30) + heavy-rain CSI points (15) + anti-blur points
  (10), computed on that set, + the shock points (15) from the event windows (shocks are only run on event windows).
- `VAL-2` Named events (last input day = 2 days before the event date):
  validation: Kerala floods 2018-08-15, cyclone Titli 2018-10-11, cyclone Gaja 2018-11-16, cyclone Fani
  2019-05-03, cyclone Vayu 2019-06-12, monsoon onset 2019-06-08, north India heat 2019-06-02;
  test: cyclone Amphan 2020-05-20, cyclone Nisarga 2020-06-03, cyclone Tauktae 2021-05-17, cyclone Yaas 2021-05-26.
- `VAL-3` Every evaluation appends one row to `runs/leaderboard.csv`.
- `VAL-4` Before submission: retrain the chosen recipe on all years 2005-2021 (phase 3).

## 4. Representation (decision B)

- `REP-1` Networks work in normalised anomaly space: `z = (f(x) - f(clim(doy))) / anom_std` (`twin/codec.py`).
- `REP-2` Networks predict the CHANGE of the anomaly for the next day (B3). B4 (residual on damped forecast) dropped.
- `REP-3` Inputs per step: 29 maps (18 anomaly maps of the 3 days, 6 target-day climatology maps, land/sea,
  height in km, sin/cos date, edge distance). Phase 2 experiment: + lat/lon maps (F6).
- `REP-4` Rain transform `f = sqrt` for `tp` (D2); identity for other channels.

## 5. Rollout and training (decision C)

- `ROL-1` Inference: autoregressive, 1 day per step, `steps` times (C1). Anomalies clamped to +-12 std per step.
- `ROL-2` Training curriculum (run 2): unroll 1 for 4 epochs, 2 for 3, 4 for 3, 8 for 2 (`1:4,2:3,4:3,8:2`).
  Weights kept from the best validation loss in the last stage. Full state saved every epoch (resumable).
- `ROL-3` Post-blend (C3): `final = (1 - w) * model + w * damped_anomaly`, weight `w[lead, channel]` in [0, 1]
  fitted by least squares on training-year rollouts. Kept only if it improves the validation score (forecast +
  heavy rain + blur parts), decided by the fitter and stored as `blend_on` in the fitted file.

## 6. Rain (decision D)

- `RAIN-1` sqrt transform (`REP-4`).
- `RAIN-2` Heavy-rain weighting (D5): on the `tp` channel, squared errors at cells where the TRUE rain > 10 mm/day
  count 3x in the training loss.
- `RAIN-3` Two-part rain head (D4), phase 2 option `--rain-head two_part`: the network outputs a 7th map, a logit
  for "rain > 1 mm/day". Amount = the usual rain output (sqrt space, >= 0). Training: rain = sigmoid(logit) x amount
  (smooth, so gradients flow) and a binary cross-entropy loss (weight 0.5) on "true rain > 1 mm". Inference:
  rain = amount where logit > 0, else 0 (hard gate, deterministic). Merge with D2 + D5 if it wins the screening.

## 7. Sharpness (decision E)

- `SHP-1` Spread loss (E2): + 0.1 x mean |std_space(pred) - std_space(true)| per channel and step.
- `SHP-2` Spectral loss (E2): + 0.05 x mean |amplitude(FFT2(pred)) - amplitude(FFT2(true))| / mean amplitude(true),
  per channel (training only; does not affect inference locality).
- `SHP-3` Variance rescale (E4), after `ROL-3`: anomaly of each non-rain channel multiplied by
  `s[lead, channel] = mean std_space(true anomaly) / mean std_space(predicted anomaly)` on training-year rollouts,
  clipped to [1, 2]. Applied per channel only where it improves the validation auto score; the choice is stored
  with the fitted factors.
- `SHP-4` Rain quantile mapping (E5), after `SHP-3`, rain only: per season (DJF, MAM, JJAS, ON) and lead day, map
  the forecast rain value at quantile q to the true rain value at quantile q (101 levels, fitted on training-year
  rollouts, piecewise linear, monotone). Kept only if validation heavy-rain skill improves.
- `SHP-5` Physics hybrid keeps advection (E6) and also gets `SHP-1..4`.
- `SHP-6` CRPS ensemble (E3): phase 2 experiment on one family.
- `SHP-8` Shock stability guard (new, phase 2 option `--shock-loss`): on every 4th training batch, also roll out the
  batch with a warm-ocean-style shock (+3 K t2m, +15% tcwv) in a random 12 x 12 box, and add
  `w x mean(relu(|shocked - base| / channel_std - 4)^2)` in physical units (judge's limit is 5 std, 4 leaves margin).
- `SHP-7` Post-processing order: blend (`ROL-3`) -> rescale (`SHP-3`) -> rain map (`SHP-4`) -> clamp (`CON-3`).
  Stored in `artifacts/post/<name>.npz`, applied by a deterministic wrapper model.

## 8. Edges (decision F)

- `EDG-1` Replicate padding + edge-distance input map (F2 + F5).
- `EDG-2` Phase 2 options. F4 `--edge-relax`: after every step (training and inference) multiply the anomaly of
  cells at distance 0, 1, 2 from the grid edge by 0.7, 0.85, 0.95 (pull toward normal), stored in the checkpoint.
  F6 `--latlon`: 2 more input maps (latitude and longitude scaled to -1..1), weight decay 1e-3 instead of 1e-4
  against overfitting. Target: the Arabian Sea shock "propagates" check.

## 9. Runtime (decision G)

- `RUN-1` Phase 3: load order ONNX (onnxruntime) -> torch `state_dict` -> damped baseline. No numpy rewrite.
- `RUN-2` torch threads = 4 (env `TWIN_THREADS`). Model loads once at import; warm-up before `/health` is ready.
- `RUN-3` Every model is deterministic: no state between calls, no unseeded randomness.

## 10. Locality (from contest shock tests)

- `LOC-1` Per-step reach must keep < 5% of a shock's response beyond 11 cells on day 1 (judge's limit).
  Measured by the reach probe; FNO and global attention are excluded for this reason.

## 11. Impact module (decision J)

- `IMP-1` Crop stress: rain deficit vs normal over land during the sowing season (June-July) plus heat above
  35 C, computed from live `/simulate` output on the dashboard side. A cropland map is a later addition (needs
  organiser OK for bundled non-weather maps). Other ideas parked in `docs/DECISIONS.md` section J.

## 12. Plan (updated in 1.1)

- **Run 2 (now):** families CNN, U-Net, CNN+transformer hybrid, physics hybrid, ConvGRU with `REP-*`, `ROL-1..2`,
  `RAIN-1..2`, `SHP-1..2`; then fit and evaluate post-processing `ROL-3`, `SHP-3`, `SHP-4` per family:
  rows `raw`, `+blend`, `+blend+rescale`, `+blend+rescale+rainmap`. Dropped: FNO (`LOC-1`), ViT, GNN, LightGBM.
- **Run 2: done** (`docs/RESULTS_RUN_2.md`). Top: U-Net, CNN+transformer hybrid, ConvGRU (tied), CNN, physics.
- **Phase 2a, screening:** U-Net (best mean) trained 4 times with the run 2 recipe plus ONE option each:
  F4 edge relax, F6 lat/lon, D4 two-part rain, SHP-8 shock guard. An option WINS if, versus `r2_unet`, the mean of
  the three validation auto scores (events, random, named) rises by >= 0.5 and shocks do not fall.
- **Phase 2b, run 3:** all 5 families (U-Net, hybrid, ConvGRU, CNN, physics) with the run 2 recipe + all winning options.
- **Phase 2c:** ensembles of the best models (equal-weight average, and with `restore_variance`), and
  `SHP-6` CRPS ensemble (E3) on the best family.
- **Phase 3:** `VAL-4` final retrain, `RUN-1` ONNX chain, brief.

## 13. Known risks

- Warming bias: 2023-24 is warmer than the 2005-2021 normal; measure the t2m bias at day 8 on test years.
- 5 hidden windows: scores are noisy; prefer improvements visible on random AND event AND named windows.
- Arabian Sea shock box touches the west edge.

## Change log

- 1.0 (2026-11-26): first version from decisions B-J (`docs/DECISIONS.md`) and sweep 1 results.
- 1.1 (2026-11-26): after run 2. Spec errors fixed: `ROL-3` blend now "keep only if it helps" (was unconditional);
  `VAL-1` now defines the per-window-set auto score. Added `SHP-8` shock stability guard; made `RAIN-3` and
  `EDG-2` concrete; plan: phase 2 screening on U-Net, then run 3 on all top 5 families.
