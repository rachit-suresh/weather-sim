# Regional / limited-area ML weather models, boundary handling, and small-data training (2022-2026)

Context for applicability: India grid 32x36 at 1 deg, daily means, 6 surface variables, ERA5 2005-2017 (~4.7k days), 1-day autoregressive steps to 10 days, CPU (4 cores) < 5 s. At inference the only inputs are the 3 previous days on this grid plus day-of-year: no boundary data. Test years 2023-24 are hidden and warmer. Locality tests: a shock stays within ~3 cells on day 1 and spreads at most 8 cells/day.

Method note: about 17 tool calls. Several primary PDFs (YingLong, the 2020 GAN/vid2vid LAM paper, Rackow et al.) could not be text-extracted, so some claims rest on search snippets. These are flagged "(snippet-level)". Statements marked "(unverified, from background knowledge)" were not checked against a source in this session.

## Q1. How do limited-area ML models get boundary information, how much does it matter, and what happens without it?

### Takeaway
Every operational or published ML limited-area model (neural-lam/Oskarsson, the DANRA/COSMO km-scale LAMs, YingLong, StormCast, the IMDAA-India benchmark, Bris/stretched-grid AIFS) gets lateral information from outside: a forced boundary ring, coarse global conditioning, or a global stretched grid. When that information is removed, published ablations report "significant degradation", with divergence after a few hours at km scale and hourly steps. None of these papers studies a model with no boundary input at all, which is our situation, so we have to handle the edges ourselves.

### Cited Findings
- **neural-lam (Oskarsson, Landelius, Lindsten 2023, Nordic MEPS):** the boundary is an input, not a prediction. At each step, predictions in the boundary area (the 10 grid nodes nearest the edge) are overwritten with ground truth. The authors observed that boundary forcing is "highly useful" for grid nodes close to the boundary. I found no quantified with/without numbers in the material retrieved. — [arXiv:2309.17370](https://arxiv.org/abs/2309.17370); [poster](https://internal.wasp-sweden.org/winterconf2024/posters/joel_oskarsson.pdf)
- The probabilistic hierarchical-GNN follow-up (Graph-EFM) uses the same boundary setup on 10 km MEPS and states that boundary conditions must be handled to model a limited domain. — [arXiv:2406.04759](https://arxiv.org/html/2406.04759v2)
- **"Building ML LAMs: km-scale forecasting in realistic settings" (2025, DANRA 2.5 km / COSMO 2.2 km):**
  - The boundary MLP takes 3 boundary times (k-1, k, k+1, so it includes future forcing). The boundary stays at its coarse native resolution. Training uses ERA5 boundaries and verification uses IFS forecasts. Loss is computed only on interior points, and mesh-to-grid edges connect only to interior points.
  - Width ablation (0/400/800/1200/1600 km): removing the boundary "significantly degrades" skill, and the model diverges from truth after a few hours (1 h steps). There is "no significant difference" among the nonzero widths, so 400 km is enough. 800 km was chosen as about 2x the fastest feature speed (~750 km per 3 h step).
  - An overlapping boundary (forcing also over the interior) helps with ERA5 forcing. With IFS forcing the gain shrinks or reverses (DANRA), which the authors attribute to over-reliance on the boundary.
  - Removing future boundary forcing gives "significant degradation". Sinusoidal boundary time-delta encodings gave no significant gain.
  - The paper gives no numeric RMSE deltas in the text, only figures 6-9.
  - Compute: 34,560 GPU-h (DANRA) and 70,656 GPU-h (COSMO) in total across runs. A 5-day forecast takes about 50-60 s on a single A100/GH200.
  - Data: 54,896 training steps (DANRA, 21 years) and 33,660 (COSMO, 4 years). The final COSMO model is fine-tuned with K=12 unrolled steps.
  - Source: [arXiv:2504.09340](https://arxiv.org/html/2504.09340v1)
- **Diffusion-LAM:** a variant without future boundary conditioning "shows a significant degradation in performance" (snippet-level, via an aggregator). — [Lacuna summary of Diffusion-LAM](https://lacuna.tiptreesystems.com/work/diffusion-lam-probabilistic-limited-area-weather-forecasting-with-diffusion/wrk_39285479aa0bc333e4ad608926917d4c)
- **Regional benchmark over India (IMDAA 0.12 deg, 256x256, 6-36.7N, 66.6-97.25E, 2000-2017 train ~26.5k six-hourly samples, 2018 validation, 2019 test):**
  - Two boundary strategies were compared: (a) a 10-pixel high-resolution boundary ring wrapped around the interior; (b) ERA5 cropped to 124x124, upsampled, and concatenated as coarse conditioning.
  - Both strategies use ground-truth (reanalysis) boundaries at evaluation. The authors list this as a limitation.
  - Under boundary forcing, Stormer and GraphCast do best. Under coarse conditioning, Stormer becomes the worst, with errors growing rapidly, while the other models do comparably or slightly better.
  - UNet is lowest-ranked but "close to the top".
  - Models are about 25-35M parameters. Results are in figures only, with no numeric RMSE in the text.
  - Source: [arXiv:2509.00653](https://arxiv.org/html/2509.00653v1)
- **YingLong (3 km regional AI LAM):**
  - LBCs come from a coarse global forecast and are "transferred to the inner area" through YingLong's own forecast on the boundary region.
  - Inserting coarse global forecasts directly at each rolling step can create discontinuities at the boundary/interior junction, which motivates their smoothing.
  - A 207 km boundary width gave the smallest RMSE among the widths tested.
  - (snippet-level; full text not extracted.) Source: [arXiv:2401.16254](https://arxiv.org/pdf/2401.16254)
- **StormCast (NVIDIA):** km-scale conditional diffusion with a U-Net, using large-scale forcing from NCEP GFS (or ERA5 in training), and targeting 0-48 h. (Snippet-level, via a review; the primary paper was not fetched.) — [review via arXiv:2603.27288](https://arxiv.org/html/2603.27288v1)
- **STCast** criticizes standard practice as "concatenating a fixed neighborhood in global area to region", which gives static, local coupling. It proposes adaptive boundary alignment. — [arXiv:2509.25210](https://arxiv.org/html/2509.25210v2)
- **Bris (MET Norway, Anemoi stretched-grid AIFS):**
  - There is no lateral boundary at all: one global graph runs at 2.5 km over the Nordics and is coarse elsewhere.
  - It is pre-trained on 43 years of 31 km ERA5, then fine-tuned on 3.3 years of 2.5 km MEPS analyses.
  - It beats the MEPS control run and ensemble mean on 2 m temperature RMSE. Winds are too strong over mountains.
  - The CRPS-trained version beats AIFS-CRPS and has run operationally 4x/day since October 2025.
  - Sources: [AIES 2026](https://journals.ametsoc.org/view/journals/aies/5/2/AIES-D-25-0001.1.xml); [arXiv:2511.23043](https://arxiv.org/html/2511.23043v2); [MET Norway seminar slides](https://www.hydrologiraadet.no/wp-content/uploads/2025/05/4-Ivar-Ambjorn-Seierstad.pdf)
- **Boundary-free LAM (older, 2020; outside the 2022-2026 window, cited for the concept only):** the forecasting problem on a limited area without prescribed boundary conditions is "ill-posed (no unique solution)". A model trained on reanalysis may partly infer boundary behaviour for "at least several hours". (Snippet-level.) — [arXiv:2006.07718](https://arxiv.org/pdf/2006.07718)

### Inferences
- **Signal travel on our grid.** Daily-mean surface signals advect at roughly 5-15 m/s, which is about 430-1300 km/day, or about 4-12 one-degree cells/day (unverified physical estimate, consistent with the 8 cells/day test cap). Boundary-unknown information can therefore contaminate ~4-12 cells inward per step. By day 3-4 most of a 32x36 domain depends on information that was outside the domain at t0. The edge problem is therefore not cosmetic. It dominates the day 3-10 error, and only climatology or trend can fill that gap.
- **The width results scale to us.** The 400 km result (~2x the per-step feature travel distance) suggests that on our grid a "relaxation ring" of ~4-8 cells (400-800 km) is the zone directly corrupted after one step.
- **We cannot use a forced ring.** Our inputs contain no future boundary, so the neural-lam design does not apply directly. The closest analogue is to treat the outer ring like neural-lam does, but fill it with a self-predicted or climatological boundary instead of truth (see Q2).
- **Architecture choice.** The India benchmark shows a simple UNet stays "close to the top" on Indian regional data. Architecture choice matters less than boundary and data handling at small scale. Note that this benchmark had ground-truth boundaries, so it does not say how UNets behave without them.

### Gaps
- No paper found that quantifies skill vs lead time for a regional ML model with zero boundary input at daily/1 deg resolution, or that tests climatology or persistence boundaries as a substitute.
- Exact RMSE deltas from the boundary ablations exist only in figures (arXiv:2504.09340 figs 6-9; Diffusion-LAM).
- MeteoSwiss/DWD regional ML: not separately researched. COSMO-2/KENDA (MeteoSwiss data) appears in 2504.09340.
- MetMamba, CorrDiff: not researched in this session. CorrDiff is a downscaling model, not an autoregressive LAM (unverified, from background knowledge).

## Q2. Techniques for a model with NO boundary inputs at inference (relaxation zones, padding, larger predicted domain, learned boundary emulators, climatology/persistence nudging)

### Takeaway
I found no direct quantitative evidence in 2022-2026 ML literature for boundary-free LAMs. The published evidence is indirect: the boundary ring only needs to be about one step's travel distance wide (400 km for 3 h steps), YingLong shows abrupt boundary insertion causes junction discontinuities, and models without future boundaries degrade significantly. For our setup the defensible design is: (a) a learned interior emulator, plus (b) an outer ring of ~3-6 cells that is blended, with weights depending on lead time and distance to the edge, toward trend-adjusted climatology (Davies-style relaxation), plus (c) loss and evaluation-aware weighting. All of these cost little on CPU.

### Cited Findings
- **Width.** The ring needs about one step's worth of feature travel (400 km was enough at 3 h steps). Wider rings gave no significant gain. — [arXiv:2504.09340](https://arxiv.org/html/2504.09340v1)
- **Smoothing.** Abrupt insertion of external boundary values at each autoregressive step causes discontinuities at the ring/interior junction. YingLong smooths this by letting its own forecast in the boundary region transfer information inward. (Snippet-level.) — [arXiv:2401.16254](https://arxiv.org/pdf/2401.16254)
- **Exclusion.** Excluding boundary points from the loss and never predicting them is the standard neural-lam practice. — [arXiv:2504.09340](https://arxiv.org/html/2504.09340v1); [arXiv:2309.17370](https://arxiv.org/abs/2309.17370)
- **Overlap trade-off.** Overlapping forcing (giving the model large-scale information over the interior too) reduces error accumulation over many autoregressive steps. With imperfect forcing (IFS instead of ERA5) the model can over-rely on it, so overlap gains shrink or reverse. — [arXiv:2504.09340](https://arxiv.org/html/2504.09340v1)
- **Coarse-context conditioning.** Adding coarse context (ERA5 over a larger area, upsampled) instead of a ring helped UNet/GraphCast-type models slightly but hurt patch-based Stormer on the India domain. — [arXiv:2509.00653](https://arxiv.org/html/2509.00653v1)

### Inferences (design options for our system, unverified unless cited)
1. **Davies-style relaxation ring toward a climatology/trend field.**
   - Rule: x_edge = w(d, t) * clim_trend + (1 - w) * model, with w ~ 1 at the outermost cell, decaying over 3-6 cells, and growing with lead day.
   - Under MSE, anomaly skill near the edges at days 2+ is close to zero without inflow information. The MSE-optimal edge prediction is therefore close to climatology (plus a slowly damped anomaly).
   - Cost: zero.
   - Locality: fine (purely local).
2. **Learn the relaxation instead of hand-tuning.** Feed the model an "edge-distance" channel and a lead-time/step-index channel. The network then learns how much to trust interior dynamics near edges, i.e. a learned boundary emulator. It can be trained by rolling out multiple steps so edge drift is penalised.
3. **Train on a larger ERA5 domain but evaluate only on 32x36.** This is training-only, because inference inputs are only 32x36, so it cannot add boundary information at test time.
   - It can still help in two ways. First, it gives more spatial samples for a translation-equivariant CNN. Second, it supports a "teacher" that sees the true boundary, distilled into a student that sees only 32x36.
   - Alternatively, train the student with randomly cropped sub-windows of the larger domain, using the same no-inflow condition. Edge behaviour then becomes a learned, position-relative skill rather than a memorised map.
4. **Padding.**
   - Zero padding in anomaly space equals "climatology outside the domain", which is physically sensible for anomalies.
   - Replicate padding equals "persistence of edge values", which promotes edge drift and stripes.
   - Reflect padding creates fake mirrored inflow.
   - For anomaly-normalised inputs, zero padding is the most defensible choice. Background reasoning only, no source found.
5. **Persistence vs climatology at edges.** Use damped persistence of the input-day anomalies, with an e-folding time of a few days for T2m/TCWV/MSLP and shorter for rain/wind. Fit the e-folding time per variable and per distance band on validation data. This is likely better than pure climatology on day 1-2 at the edges. Cost: trivial.
6. **Locality constraints.**
   - A 3x3-kernel CNN of depth L has a receptive field of L cells each way.
   - To respect "within ~3 cells on day 1" and "<= 8 cells/day", per-step receptive field radius must be <= 3 (strict) or <= 8 (loose). Any global operation (attention, FFT/FNO, global pooling, a large U-Net bottleneck) on the state violates the shock-locality test unless masked. Stormer/ViT-style and FNO-style models are excluded or need windowed attention.
   - This favours shallow local CNNs or local GNNs, which the India benchmark suggests are competitive.

### Gaps
- No quantitative evidence found for climatology/persistence boundary nudging in ML LAMs, for learned boundary emulators without external data, or for padding-type comparisons. These would have to be tested empirically (cheap at our scale).
- No published day-1 to day-10 "edge error vs distance-from-edge" curves for an ML LAM without boundary forcing were found.

## Q3. Data-efficient training with ~4.7k samples (transfer from global pretrained models, more years/larger region, augmentation, multi-step objectives, regularisation, size scaling)

### Takeaway
The best-documented route is **pre-train on long, coarse ERA5 history, then fine-tune on the small target set**. Bris does this (43 years of ERA5, then 3.3 years of regional analyses) and beats regional NWP on T2m RMSE. Fine-tuning with multi-step unrolled rollouts is standard. Large foundation models (Aurora 1.3B, Prithvi WxC 2.3B) can be fine-tuned regionally but cannot run under 5 s on 4 CPU cores. They are useful at most as offline teachers or data generators. For us, the single highest-value step is to **use all permitted ERA5 (1979/1940-2022) instead of 2005-2017**, which gives 3-15x more daily samples, plus training on a larger surrounding region.

### Cited Findings
- **Bris:** pre-training on 43 years of 31 km ERA5, then fine-tuning on only 3.3 years of 2.5 km MEPS analyses, yields a regional model better than MEPS control/ensemble mean for T2m RMSE. — [AIES 2026](https://journals.ametsoc.org/view/journals/aies/5/2/AIES-D-25-0001.1.xml); [MET Norway slides](https://www.hydrologiraadet.no/wp-content/uploads/2025/05/4-Ivar-Ambjorn-Seierstad.pdf)
- **km-scale LAMs (DANRA):** 80 pre-training epochs, then 3 epochs of fine-tuning with 4-step rollouts. COSMO gets a further K=12-step unrolled fine-tune to stay skilful to 120 h. — [arXiv:2504.09340](https://arxiv.org/html/2504.09340v1)
- **Prithvi WxC (2.3B parameters, MERRA-2, 160 variables):** for regional downscaling, the encoder and decoder were frozen and a U-Net-style component trained. The paper reports >4x improvement over interpolation baselines. — [arXiv:2409.13598](https://arxiv.org/pdf/2409.13598); [GitHub](https://github.com/NASA-IMPACT/Prithvi-WxC)
- **Parameter-efficient fine-tuning:** Aurora (1.3B) and Prithvi WxC were evaluated for regional precipitation over China. Prithvi was harder to fine-tune because its pretraining data was more dissimilar. — [arXiv:2509.22020 / ICLR 2026](https://arxiv.org/pdf/2509.22020)
- **India benchmark:** used ~26.5k six-hourly samples (18 years), 25-35M-parameter models, 100 epochs, AdamW. UNet was close to the best GNN/transformer models. — [arXiv:2509.00653](https://arxiv.org/html/2509.00653v1)
- **Climate-emulator transfer:** fine-tuning on only small amounts of data from a distribution similar to the test set enabled good performance in a warmed climate. Generalization "is not an intrinsic feature" of data-driven emulators. — [arXiv:2405.18585](https://arxiv.org/html/2405.18585)

### Inferences
- **More years is the cheapest big win.** Training data "may use any public data from before 2023". ERA5 1979-2022 gives ~16k daily samples (3.4x). ERA5 back-extension from 1940 gives ~30k (6x).
  - Recent years 2018-2022 are both allowed and closest to the warm test period. They should be in training, or used for the final fine-tune and validation, rather than leaving the model on 2005-2017.
  - Caveat: pre-1979 ERA5 has fewer observations, so quality is lower. Weight it less or use it only for pre-training.
- **Larger region pre-training.** A translation-equivariant local CNN can be pre-trained on many 32x36 (or larger) windows from a wider ERA5 box (e.g. 20S-50N, 30-130E), then fine-tuned on the target box with position/static channels (orography, land-sea mask, lat).
  - This multiplies effective samples by the number of non-redundant windows. It also teaches inflow behaviour, because a window's interior near its edges sees neighbours. This matches the pre-train/fine-tune pattern that worked for Bris.
- **Augmentation.**
  - Physically valid: random crops and translations of windows on the larger domain (with lat/orography channels), and time-jittered start dates.
  - Not physically valid: horizontal flips and rotations. They break Coriolis/beta and monsoon geometry, and flips require sign changes of u/v. Avoid them, or use them only with careful vector handling and a hemisphere caveat. Background reasoning, no source found.
- **Multi-step objectives.** Unrolled fine-tuning (4-12 steps in the cited LAMs) is the standard fix for autoregressive drift. For daily steps, unroll 3-5 days in the final phase, with gradients through the rollout. That is cheap for a small CNN.
- **Model size.** With ~5-30k samples and a CPU budget, ~0.1-5M-parameter local CNNs are appropriate. The 25-35M models in the India benchmark used 26.5k samples on a 256x256 grid, which is roughly 2,000x more pixels per sample than our 32x36. Use weight decay (AdamW), early stopping on held-out years, and ensembling of a few seeds. Ensembling is cheap and reduces variance. Unverified general practice.
- **Foundation models.** Aurora/Prithvi/GraphCast inference at 1 deg global would far exceed 5 s on 4 CPU cores (unverified, but they are 1B+ parameter models). Use them only offline, e.g. as feature teachers. That is probably not worth it given the 2022 data cutoff and our tiny grid.

### Gaps
- No paper found that reports sample-size scaling curves (skill vs number of training years) for regional daily emulators.
- GraphCast and FourCastNet fine-tuning-to-region numbers were not retrieved.
- ClimaX regional numbers were not retrieved (ClimaX had a regional forecasting task over North America in its 2023 paper; unverified here).
- No source found on which augmentations are physically valid for gridded weather.

## Q4. Non-stationarity / warming between training years and test years

### Takeaway
ML weather and climate models trained on historical data are systematically cold-biased in later, warmer periods, behaving like climates **15-20 years earlier** (up to 20-30 years in some regions). The bias is worst at the hottest temperatures. Models without forcing inputs drift toward the training climate. Remedies with evidence:
- condition on external forcing (ACE2 uses CO2/SST forcing);
- fine-tune on a small amount of data close to the test distribution.

For us, add trend-adjusted climatology and train through 2022.

### Cited Findings
- **Landsberg & Barnes (GRL 2026):** FourCastNet, Pangu and ACE2 all produce cold-biased boreal-winter land temperatures, resembling climates 15-20 years before the prediction period, and up to 20-30 years in the Eastern US. FourCastNet and Pangu are most cold-biased for the hottest predictions, which points to limited exposure to modern extremes. ACE2's bias is largest where climate change is most pronounced. — [arXiv:2509.22359](https://arxiv.org/html/2509.22359v2); [GRL listing](https://doaj.org/article/eb2b008003ba408cb97c45ef60fe5bfa)
- **Rackow et al. (2024):**
  - Training periods: Pangu on ERA5 1979-2017; GraphCast 1979-2017, then fine-tuned on 2016-2021 operational data; AIFS 1979-2018, then fine-tuned on 2019-2020.
  - Global-mean T2m drift over 10-day forecasts: Pangu -0.03 K/day in 2023 and -0.07 K/day in the +2.9 K (2049) climate. GraphCast shows little 2023 bias but cancelling errors (land cooling vs tropical ocean warming). AIFS has small 2023 bias and -0.04 K/day in 2049.
  - The authors attribute the drift to missing ocean, sea-ice and land inputs, so models relax toward the training climate. They suggest adding forcing information and sea-ice/soil-moisture inputs; these were not tested.
  - Source: [arXiv:2409.18529](https://arxiv.org/html/2409.18529)
- **ACE2:** 450M parameters, 6-hourly, 1 deg, with CO2/SST forcing inputs. ACE2-SOM with a slab ocean reproduces global annual-mean surface trends under +2%/yr CO2 and equilibrium responses to unseen CO2 levels. Transient and abrupt changes are harder. — [arXiv:2412.04418](https://arxiv.org/html/2412.04418v2); [Ai2 blog](https://allenai.org/blog/ai2-climate-emulator)
- ACE fails when SST and CO2 are inconsistent with each other, and ACE2-SOM under-performs for abrupt CO2 change. — [arXiv:2606.07928](https://arxiv.org/html/2606.07928)
- Because SST and CO2 co-vary historically, emulators can attribute change to the wrong forcing. ReRoute adds a fixed radiative pathway and improves out-of-distribution skill with light fine-tuning. — [arXiv:2610.02252](https://arxiv.org/pdf/2610.02252)
- Small-sample fine-tuning on data near the test distribution enables good performance in a warmed climate. — [arXiv:2405.18585](https://arxiv.org/html/2405.18585)

### Inferences
- **Why our model will run cold in 2023-24.** It predicts anomaly changes relative to a 2005-2017 climatology, and any damping pulls toward that climatology. The cited studies show exactly this mechanism, 10-day drift toward the training climate. Over India, 2023-24 is likely ~0.3-0.6 K above a 2005-2017 mean (unverified magnitude; compute it from ERA5 1979-2022 trends).
- **Mitigations (all CPU-free at inference):**
  - (a) Build the climatology as a per-cell, per-day-of-year linear (or LOWESS) trend fit over 1979-2022, extrapolated to the forecast year. That needs a year/time input, but the test date is known from day-of-year plus the input dates. If the year is truly not given, use (c).
  - (b) Train on data through 2022 and up-weight or fine-tune on 2015-2022.
  - (c) **Input-anchored offset.** Estimate the current "climate offset" from the 3 input days, e.g. the domain-mean T2m anomaly, low-pass filtered. Let the model relax toward climatology + offset rather than raw climatology. This acts as an implicit "global-mean-temperature conditioning" in the spirit of ACE's forcing input, without needing external data.
  - (d) Avoid strong anomaly damping to zero for T2m/TCWV. Damp toward the anchored offset instead.
- **Variable sensitivity.** TCWV also trends upward with warming (~7%/K, Clausius-Clapeyron; background knowledge), so the same treatment applies. MSLP and winds are less trend-affected.
- **Validation.** Use the latest allowed years (e.g. 2021-2022) as validation to detect the trend bias before the hidden test.

### Gaps
- No regional (India) quantification found of the 2023-24 vs 2005-2017 T2m/TCWV offset in ERA5. This must be computed from data.
- No paper found that tests "year" or "global-mean-temperature" conditioning for short-range regional emulators specifically. ACE2 conditions on CO2/SST for climate runs, not on a global-mean-temperature scalar.
