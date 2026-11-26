# weather-sim specification

**Version:** 1.14 (2026-11-27). **Status:** active.
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
- `CON-7` (organisers, 2026-11-26) Weights are committed as **plain git files, not Git LFS pointers**: the repo is run
  offline and nothing is downloaded. `.gitattributes` has no LFS rules.
- `CON-8` (organisers) `artifacts/active.json` points at a real trained model (never the damped baseline or the
  starter's placeholder). The kit's schema validator passes against `python serve.py --port 8000` before every push.
- `CON-9` (organisers) The 2-page brief is a PDF in the repo: `BRIEF.pdf` at the root. Deadline: 2026-11-27 10:00.
- `CON-10` (organisers: "a README on how to train and run") `README.md` gives the exact commands to (1) run the server and
  dashboard offline, (2) rebuild the artifacts and reproduce the SUBMITTED model from the training data, (3) run the
  checks. No Git LFS instructions (`CON-7`). Release freeze: nothing is merged to `main` after 2026-11-27 08:30.

## 2. Architecture

- `ARC-1` `serve.py` (HTTP only) -> `twin/engine.py` (shock, clamps, fallback) -> active model from
  `artifacts/active.json`. Server-side code lives only in `serve.py` and `twin/`.
- `ARC-2` Training, fitting and evaluation live in `training/` and are never imported by the server.
- `ARC-3` Crash fallback: if the active model fails to LOAD, or raises / returns non-finite values while
  forecasting, the damped-anomaly baseline answers (deterministic). The server must always start.
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
- `VAL-4` Final model: the chosen recipe (`r3_cnn`: CNN, run 2 recipe + `RAIN-3`, seed 0) retrained on ALL years
  2005-2021 (`--train-years 2005-2021 --final`): anomaly scales from all years, same curriculum, and the weights after
  the LAST epoch are kept (no held-out years remain to select on). Acceptance before it is served: kit validator
  passes, zero shock == null, valid outputs, reach probe local (day-1 max reach <= 11 cells), mean auto score on
  2018-19 and 2020-21 windows (now in-sample) >= `r3_cnn`'s minus 0.5, Amphan and June-2019 cases sane. If any check
  fails, `r3_cnn` stays the served model.
  **Result: all checks passed; `final_cnn` is served** (`docs/RESULTS_RUN_3.md`).

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
  for "rain > 1 mm/day". Amount = the usual rain output (sqrt space, >= 0). Rain = amount where logit > 0, else 0
  (hard gate), in BOTH training and inference, so the rollout always sees the same kind of rain. Training passes
  gradients through the gate as if it were sigmoid(logit) (straight-through estimator), plus a binary
  cross-entropy loss (weight 0.5) on "true rain > 1 mm". Merge with D2 + D5 if it wins the screening.

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

- `RUN-1` ONNX export: **dropped** (team decision 2026-11-27, too much change this late). Load order: torch
  `state_dict` -> damped baseline (`ARC-3`). No numpy rewrite.
- `RUN-4` torch-version tolerance: load checkpoints with `weights_only=True` and, if the installed torch does not
  support that argument (torch < 1.13), load again without it. Checkpoints hold only tensors and plain Python values.
- `RUN-2` torch threads = 4 (env `TWIN_THREADS`). Model loads once at import; warm-up before `/health` is ready.
- `RUN-3` Every model is deterministic: no state between calls, no unseeded randomness.

## 10. Locality (from contest shock tests)

- `LOC-1` Per-step reach must keep < 5% of a shock's response beyond 11 cells on day 1 (judge's limit).
  Measured by the reach probe; FNO and global attention are excluded for this reason.

## 11. Impact module (decision J)

- `IMP-1` Impact module = three "lenses" computed in the dashboard (`frontend/src/lib/impact.ts`) from every live Prime
  and Shocked rollout: (1) flood and extreme-rain risk index (rain intensity, column moisture, terrain slope; IMD heavy
  64.5 / very heavy 115.6 mm/day); (2) heatwave stress as Stull wet-bulb temperature from t2m and humidity inferred from
  water vapour (31 / 33 / 35 C thresholds); (3) crop water balance (rain minus Hargreaves evapotranspiration, stress
  below -2 mm/day) plus wind power density and a solar cloud proxy. Each lens shows formula, thresholds and caveats.
- `IMP-2` Dashboard (`dashboard/` built from `frontend/`): globe and flat pixel-grid views, shock brush and region
  presets, branch timeline (Prime vs Shocked), station inspector, impact rail; offline (no CDN, fonts embedded).
- `IMP-3` Art style v2 (branch `feature/pixel-art-india`; merged into main ONLY with explicit team approval):
  (a) an India art map generated in the browser at 10 art pixels per degree: coastline traced from the model's own
  land/sea mask (art matches forecast cells), white surf line, brown extruded edge on the south/east coasts, stacked
  grey peak sprites from elevation plus hand-drawn ranges (Himalaya, Western/Eastern Ghats, Aravalli, Vindhya-Satpura),
  tree clusters (north-east, Western Ghats, central India), Thar dunes, 8 major rivers, teal sea with wave marks,
  boats and sea labels, neighbouring countries in muted parchment;
  (b) flat map draws the art map directly; (c) globe magnifies India 5x (focus warp k = 5, was 2.65) and samples the
  art map inside the grid; (d) "Storybook" overlay (default): translucent pixel tint of the forecast field plus weather
  sprites (clouds/rain where wet, sun where hot, wind arrows); "Data" toggle restores the full colour field;
  (e) no new network dependencies, no change to the server API, rollout must stay live (`IMP-2`);
  (f) a faint 32 x 36 cell grid is drawn over the art map (flat and globe) so the forecast cells stay visible;
  (g) "Weather focus" slider w in [0, 1] (default 0): the art is faded toward paper by up to 60% and the forecast tint
  opacity moves from the storybook value to 0.85 for every cell, so a demo can dull the background and show the weather;
  the coast and India border are inked over the forecast (fading in with w) so the geography stays readable.
  (h) geography of the art map (1.14, replaces the "traced from the model's land/sea mask" part of (a)): coastline
  and islands (Andaman & Nicobar, Lakshadweep, Sri Lanka) come from hand-entered lat/lon outlines at about 0.2 deg,
  India vs neighbour from India's official outline (whole of Jammu & Kashmir and Ladakh, as on Survey of India maps);
  biome edges are noise-warped so no straight 1-deg lines show. Only the picture changes: forecast cells, masks used for
  metrics, and the API are untouched.
- `IMP-4` Uncluttered layout (1.14): on load the page shows only the slim header, the map and a one-line day slider.
  (a) Shocks live in a left drawer opened by a "Shocks" button: judge shocks first, "More presets" and "Paint your own"
  folded; the queued shocks and the BRANCH TIMELINE button stay at the drawer bottom, and a floating BRANCH button
  appears on the map when shocks are queued; (b) results live in a right drawer opened by an "Impact" button (it shows a
  dot when a new branch is ready; clicking a cell opens it on the cell inspector); (c) the six variable buttons become
  one dropdown with plain names; display settings (wind, cities, storybook, weather focus, focus region) move to a
  "View" dropdown; (d) the daily chart is folded under the slider; (e) a "?" button opens a short how-it-works modal;
  (f) scenario labels are plain ("Monsoon onset, 13 May 2020"); run id and latency move into the model badge tooltip;
  (g) view names stay Prime / Branched / Diff (team choice); every existing feature stays reachable.
## 12. Plan (updated in 1.1)

- **Run 2 (now):** families CNN, U-Net, CNN+transformer hybrid, physics hybrid, ConvGRU with `REP-*`, `ROL-1..2`,
  `RAIN-1..2`, `SHP-1..2`; then fit and evaluate post-processing `ROL-3`, `SHP-3`, `SHP-4` per family:
  rows `raw`, `+blend`, `+blend+rescale`, `+blend+rescale+rainmap`. Dropped: FNO (`LOC-1`), ViT, GNN, LightGBM.
- **Run 2: done** (`docs/RESULTS_RUN_2.md`). Top: U-Net, CNN+transformer hybrid, ConvGRU (tied), CNN, physics.
- **Phase 2a, screening:** U-Net (best mean) trained 4 times with the run 2 recipe plus ONE option each:
  F4 edge relax, F6 lat/lon, D4 two-part rain, SHP-8 shock guard. An option WINS if, versus `r2_unet`, the mean of
  the three validation auto scores (events, random, named) rises by >= 0.5 and shocks do not fall.
- **Phase 2a: done.** Versus r2_unet (mean 66.51, shocks 13.12): D4 two-part rain head with straight-through gate
  67.21 / 13.88 -> **WINS**; edge relax 66.98 / 13.12 (+0.47, just under the bar); lat/lon 66.87 / 13.12;
  shock guard 66.07 / 13.12. None of the edge options fixed the Arabian Sea "spreads outward" check.
- **Phase 2b, run 3:** all 5 families (U-Net, hybrid, ConvGRU, CNN, physics) with the run 2 recipe + `RAIN-3`
  (`--rain-head two_part`). Tags `r3_<family>`. U-Net is trained too (team decision), although `p2_unet_d4st` used
  the same recipe, options and seed: `r3_unet` doubles as a reproducibility check (scores should match closely).
- **Phase 2b+, D4 + edge relax (approved by the team):** U-Net with `--rain-head two_part --edge-relax`
  (`p2_unet_d4edge`) versus `p2_unet_d4st`, same win rule (mean of the 3 window sets +0.5, shocks not lower).
  If it wins, run 3b adds edge relax to the other families.
  **Result: WINS.** p2_unet_d4edge 66.20 / 68.70 / 69.25 (mean 68.05, +0.84) with shocks 14.25 (vs 13.88).
- **Phase 2b, run 3b:** CNN, hybrid, ConvGRU, physics with run 2 recipe + `RAIN-3` + `EDG-2` F4 edge relax
  (tags `r3b_<family>`); U-Net's run 3b model is `p2_unet_d4edge`. Run 3 (D4 only) keeps running as the
  per-family comparison.
- **Submission (decided):** `r3_cnn` (run 3 CNN, D4 rain head). Chosen on the untouched 2020-21 test years:
  test mean 68.0 vs 66.5-67.0 for the other top models (`docs/RESULTS_RUN_3.md`). Ensembles and run 3b did not beat it there.
- **Phase 2d, CNN + physics blend (team request, after the Amphan walkthrough):** the physics hybrid tracks cyclones
  (Amphan landfall centre 19.5N vs real 20.5N; CNN stays at 13.5N) but scores lower overall. Blends of `r3_cnn` and
  `r3_physics` via `BlendEnsemble`: (a) 50/50 all channels; (b) physics for msl, u10, v10 and CNN for t2m, tp, tcwv;
  (c) 50/50 for msl, u10, v10, CNN for the rest. A blend replaces `r3_cnn` only if its mean of the three window sets is
  >= r3_cnn's on BOTH validation (67.92) and test (67.96) years, shocks are not lower, and the Amphan track improves.
  **Result: no blend wins; `r3_cnn` stays.** All three move Amphan to the coast (19.5N at landfall), but on the test
  years they score 67.38 (a), 66.10 (b), 67.00 (c) with shocks 12.0-12.4 vs 13.12. (b) is best on validation (68.29).
- **Phase 2c:** ensembles of the best models (equal-weight average, and with `restore_variance`), and
  `SHP-6` CRPS ensemble (E3) on the best family.
- **Phase 3 (final release v1.1):** `VAL-4` final retrain + acceptance, `RUN-4`, `CON-10` README, brief says the model
  is trained on 2005-2021, final gate on a fresh clone (validator, repo scan 15/15, tests, dashboard live), tag `v1.1`.

## 13. Known risks

- Warming bias: 2023-24 is warmer than the 2005-2021 normal; measure the t2m bias at day 8 on test years.
- 5 hidden windows: scores are noisy; prefer improvements visible on random AND event AND named windows.
- Arabian Sea shock box touches the west edge.

## Change log

- 1.0 (2026-11-26): first version from decisions B-J (`docs/DECISIONS.md`) and sweep 1 results.
- 1.1 (2026-11-26): after run 2. Spec errors fixed: `ROL-3` blend now "keep only if it helps" (was unconditional);
  `VAL-1` now defines the per-window-set auto score. Added `SHP-8` shock stability guard; made `RAIN-3` and
  `EDG-2` concrete; plan: phase 2 screening on U-Net, then run 3 on all top 5 families.
- 1.2 (2026-11-26): spec error in `RAIN-3`: soft gate in training but hard gate at inference made the forecast
  rain too high and the 10-day rollout drift (screening run p2_unet_d4: raw 38/22/40 vs 63/68/68). Now the hard gate
  is used in both, with straight-through gradients. D4 screening is rerun as p2_unet_d4st.
- 1.3 (2026-11-26): phase 2a results recorded; run 3 = run 2 recipe + `RAIN-3` on all 5 families.
- 1.4 (2026-11-26): D4 + edge relax screening added; r3_unet reuses p2_unet_d4st (identical run).
- 1.5 (2026-11-26): team asked to keep training r3_unet; it now serves as a reproducibility check.
- 1.6 (2026-11-26): D4 + edge relax wins the screening; run 3b adds it to the other 4 families.
- 1.14 (2026-11-27): spec error in `IMP-3a`: a 1-deg mask cannot draw real coasts (no Andaman Islands, fake land in
  the Andaman Sea next to Port Blair, India cut off in Kashmir, square biome patches near Jharkhand); new `IMP-3h`
  (art geography from lat/lon outlines). New `IMP-4` uncluttered layout (team request).
- 1.13 (2026-11-27): `IMP-3` art style v2 (pixel-art India map, 5x globe focus, storybook overlay), team-approved plan.
  Amended on the branch: `IMP-3f` faint cell grid, `IMP-3g` weather-focus slider (team request).
- 1.12 (2026-11-27): final release plan approved: `VAL-4` made concrete (all years, last epoch, acceptance checks);
  `RUN-1` ONNX dropped; new `RUN-4` torch tolerance and `CON-10` README + release freeze. Spec gap found: README
  requirement was missing, so the README still described Git LFS.
- 1.11 (2026-11-26): phase 2d result recorded: no blend replaces r3_cnn.
- 1.10 (2026-11-26): phase 2d CNN + physics blend experiment and its acceptance rule.
- 1.9 (2026-11-26): spec error: `IMP-1` described a rain-deficit crop panel, but the team's merged dashboard implements
  three impact lenses; spec updated to the product (`IMP-1`, new `IMP-2`).
- 1.8 (2026-11-26): run 3 / 3b / ensembles / test-year check recorded; submission model `r3_cnn`; brief built by
  `docs/brief/make_brief.py` into `BRIEF.pdf`.
- 1.7 (2026-11-26): organiser checklist: `CON-7` plain-git weights, `CON-8` real active model + validator,
  `CON-9` brief PDF. Spec error fixed: `ARC-3` did not cover a model that fails to load (server would not start).
