# Global ML weather prediction models (2022-2026): lessons for a small regional daily-mean emulator

Context for applicability judgements: our system is a 32x36 grid of 1-degree cells over India, 6 surface daily-mean variables, about 4,700 ERA5 training days (2005-2017), 1-day autoregressive steps to 10 days, 4 CPU cores and under 5 s per rollout, and a 239k-parameter local residual CNN. It is graded on RMSE skill at days 1, 4 and 8, heavy-rain CSI, day-8 spatial variance, and locality/causality shock tests (a shock spreads at most 8 cells/day, never exceeds 5 std, and a zero shock gives identical output).

Research note: I used 24 search and fetch calls. Where I could reach only an abstract, I say so. Items I recalled from memory but did not re-verify this session are marked "(unverified)" or listed under Gaps.

## Q1. Core idea of each model (architecture, time step, loss, multi-step fine-tuning, ensembles)

### Takeaway
The successful global models share one recipe regardless of backbone. They predict a normalised residual (tendency) rather than the full state, use loss weights by area, variable and level, fine-tune on multi-step autoregressive rollouts (usually up to 12 steps, about 3 days), and increasingly replace MSE with ensemble or probabilistic losses (diffusion, CRPS) so fields stay sharp. Backbones vary (mesh GNN, Swin/3D transformer, plain ViT, spherical Fourier operator), and the ablations suggest backbone choice matters less than the training recipe.

### Cited Findings
**GraphCast (DeepMind; Lam et al., arXiv 2022 / Science 2023)**
- Global 0.25-degree, 10-day forecasts of hundreds of variables, produced in under one minute — [arXiv 2212.12794](https://arxiv.org/abs/2212.12794)
- Encode-process-decode GNN with 36.7M parameters. The "multimesh" is an icosahedron refined 6 times (finest level 40,962 nodes), and edges from all refinement levels are kept, so one message-passing step can move information thousands of km — [arXiv PDF](https://arxiv.org/pdf/2212.12794)
- Autoregressive steps in the loss rise from 1 to 12 (6 h to 3 days) during training, with backpropagation through time. Training took about 4 weeks on 32 Cloud TPU v4 — [arXiv PDF](https://arxiv.org/pdf/2212.12794); summary corroborated by [third-party summary](https://gist.github.com/wolfram77/fffffc414960fc9d3dab3bddd9163a8a)
- Surface loss weights reused later by Stormer: T2m 1.0; U10, V10 and MSLP 0.1 each — [Stormer, arXiv 2312.03876](https://arxiv.org/html/2312.03876)

**GenCast (DeepMind; Price et al., arXiv 2023 / Nature 2024)**
- Conditional diffusion model at 0.25 degrees with 12 h steps, giving 15-day ensembles of more than 80 variables — [arXiv 2312.15796v2](https://arxiv.org/html/2312.15796v2)
- Encoder maps the grid to a 6-times-refined icosahedral mesh. The processor is a graph transformer in which each node attends to its k-hop neighbourhood. The model samples a residual relative to the previous state, normalised to unit variance per variable and level — [arXiv 2312.15796v2](https://arxiv.org/html/2312.15796v2)
- Trained on ERA5 1979-2018. The sampler is DPMSolver++2S with 20 steps (39 denoiser evaluations) per 12 h step. Ensembles have 50 members — [arXiv 2312.15796v2](https://arxiv.org/html/2312.15796v2)

**WeatherNext 2 / FGN (Google DeepMind and Google Research, 2025)**
- Functional Generative Network, trained directly to minimise per-location CRPS on ERA5 plus operational HRES initial conditions. It produces a 64-member ensemble in one forward pass, with 8x faster inference than diffusion-based GenCast — [arXiv 2506.10772](https://arxiv.org/html/2506.10772v1); [Google blog](https://blog.google/technology/google-deepmind/weathernext-2/?linkId=27987866)
- Google's developer guide now recommends "WeatherNext 3" for new projects. I found no paper for WeatherNext 3 — [Google developers](https://developers.google.com/weathernext/guides/models-wn2)

**Pangu-Weather (Huawei; Bi et al., arXiv 2022 / Nature 2023)**
- 3D Earth-specific transformer. "Hierarchical temporal aggregation" trains separate models for 1 h, 3 h, 6 h and 24 h steps, and at inference a greedy scheme always uses the longest step that fits, which cuts the number of autoregressive steps. Trained on 39 years of ERA5. Inference takes about 1.4 s on one GPU — [Nature PDF](https://www.nature.com/articles/s41586-023-06185-3.pdf); [arXiv 2211.02556](https://arxiv.org/pdf/2211.02556)

**FuXi (Fudan; Chen et al., npj Clim. Atmos. Sci. 2023)**
- A cascade of models fine-tuned for successive lead-time windows (15-day system). The authors report that a single base model "deteriorates significantly as lead times increase, primarily due to accumulation error" — [arXiv 2306.12873](https://arxiv.org/pdf/2306.12873); [npj](https://www.nature.com/articles/s41612-023-00512-1.pdf)

**FengWu (Shanghai AI Lab; Chen et al., 2023)**
- Multi-modal, multi-task design: modality-specific encoder-decoders feed a cross-modal fusion transformer. An "uncertainty loss" balances variables in a region-adaptive way, and a replay buffer improves medium-range performance. 0.25 degrees, 37 levels, 39 years of ERA5 — [arXiv 2304.02948](https://arxiv.org/abs/2304.02948)

**FourCastNet / SFNO (NVIDIA; Pathak et al. 2022; Bonev et al. ICML 2023)**
- SFNO replaces the flat-geometry discrete Fourier transform in FNO/AFNO with spherical harmonic transforms, avoiding artefacts and dissipation on the sphere. It achieves "stable auto-regressive rollouts for a year of simulated time (1,460 steps)" — [arXiv 2306.03838](https://arxiv.org/abs/2306.03838)
- FourCastNet (AFNO) 5-day Z500 RMSE as quoted in the Pangu paper: 484.5 for a single model and 462.5 for a 100-member ensemble — [Pangu arXiv](https://arxiv.org/pdf/2211.02556)

**ECMWF AIFS (deterministic; Lang et al. 2024)**
- GNN encoder/decoder (attention variant) with a 16-layer pre-norm transformer processor using shifted-window attention along latitude bands. Encoder and decoder run on N320 (542,080 points, about 31 km); the processor runs on O96 (40,320 points). 6 h step — [arXiv 2406.01465](https://arxiv.org/html/2406.01465)
- Training stages: (1) pre-train on ERA5 1979-2020 for 260k steps; (2) rollout training on ERA5 1979-2018, adding one step every 1,000 steps up to 72 h (12 steps); (3) a further rollout fine-tune on IFS operational analyses 2019-2020. Loss is area-weighted MSE with per-variable scaling; vertical velocity is down-weighted and weights fall linearly with height — [arXiv 2406.01465](https://arxiv.org/html/2406.01465)

**AIFS-CRPS / AIFS-ENS (Lang et al. 2024)**
- Same encoder-processor-decoder design. The processor is a 16-layer sliding-window transformer (dim 1024, 229M parameters). Each member gets independent Gaussian noise, passed through an MLP, which drives conditional layer norms in the processor — [arXiv 2412.15832](https://arxiv.org/html/2412.15832)
- Loss is "almost fair" CRPS: afCRPS_alpha = alpha·fCRPS + (1-alpha)·CRPS, with alpha = 0.95. Training stages: one-step for 300k iterations, then 2-step for 60k, then 3-to-12-step rollouts for about 45k, then a fine-tune on IFS analyses. Training uses 4 members (O96) or 2 members (N320) — [arXiv 2412.15832](https://arxiv.org/html/2412.15832)

**Stormer (Nguyen et al., 2023)**
- Plain ViT backbone with a weather-specific embedding: each variable is tokenised separately and combined by one cross-attention layer. Lead time enters through adaptive layer norm. Data is WeatherBench 2 ERA5 at 1.40625 degrees (128x256) — [arXiv 2312.03876](https://arxiv.org/html/2312.03876)
- "Randomized dynamics forecasting": the model predicts the change X_dt - X_0, with dt drawn uniformly from {6, 12, 24} h. At inference, forecasts from many interval combinations reaching the same lead time are averaged. Loss is pressure-weighted (near-surface levels weigh more) and latitude-weighted. Multi-step fine-tuning runs K=1 for 100 epochs, then K=4 for 20, then K=8 for 20 — [arXiv 2312.03876](https://arxiv.org/html/2312.03876)

**ArchesWeather and ArchesWeatherGen (Couairon et al. 2024)**
- 3D Swin U-Net (Pangu-like) at 1.5 degrees with a 24 h step. The 3D local attention is replaced by 2D horizontal windows plus "Cross-Level Attention" (column-wise attention over all levels). An upsampling-plus-convolution head reduces checkerboard artefacts. Hour and month enter through adaptive layer norm. Sizes: S 49M, M 89M, L 164M — [arXiv 2405.14527](https://arxiv.org/html/2405.14527)
- Late training on recent years only (2007-2018) to reduce distribution shift, then 20k steps of 2-4-step rollout fine-tuning. Ensembles are averages of independently seeded models (Mx4) — [arXiv 2405.14527](https://arxiv.org/html/2405.14527)
- ArchesWeatherGen trains a flow-matching generative model on top of the deterministic ArchesWeather to map its predictions onto the ERA5 distribution (1.5 degrees) — [arXiv 2412.12971](https://arxiv.org/abs/2412.12971). I could not confirm from the abstract that it models the residual of the deterministic forecast.

**ACE2 (Ai2; Watt-Meyer et al. 2024)**
- SFNO (embedding 384), about 450M parameters, 1-degree Gaussian grid, 8 vertical layers, 6 h step. Forcings: SST, global CO2, TOA solar, land/ocean/sea-ice fractions, topography. Outputs include T2m, 10 m winds, surface precipitation and diagnostic fluxes — [arXiv 2411.11268](https://arxiv.org/html/2411.11268)
- A "physics corrector" applies hard constraints: exact global dry-air-mass and column-moisture conservation, and non-negativity of moisture, precipitation and radiative fluxes. The loss is MSE summed over 2 autoregressive steps with per-variable weights. Checkpoints were selected by climate skill — [arXiv 2411.11268](https://arxiv.org/html/2411.11268)

**Aurora (Microsoft; Bodnar et al., arXiv 2024 / Nature 2025)**
- A 3D Perceiver encoder maps heterogeneous inputs into a latent with 3 compressed levels. The backbone is a 48-layer 3D Swin U-Net, followed by a Perceiver decoder; 1.3B parameters. Rollout fine-tuning uses LoRA on the attention layers, the "pushforward trick" (gradient only through the last rollout step), and a replay buffer — [arXiv 2405.13063](https://arxiv.org/html/2405.13063)

**Prithvi WxC (NASA/IBM; Schmude et al. 2024)**
- 2.3B-parameter encoder-decoder transformer trained on 160 MERRA-2 variables with a mixed objective of masked reconstruction and forecasting. Fine-tuned for rollout forecasting, downscaling, gravity-wave flux parameterisation and extremes — [arXiv 2409.13598](https://arxiv.org/abs/2409.13598)

**Aardvark Weather (Cambridge/Turing; Vaughan et al., Nature 2025)**
- End-to-end system from raw observations: SetConv plus ViT encoder onto a 1.41-degree grid; a processor of 10 ViTs with **24 h steps** applied autoregressively to 10 days; per-lead-time decoders (small UNet plus SetConv plus MLP) to stations. Stages: pre-train on ERA5, fine-tune the processor sequentially on the encoder's own outputs (pushforward style), then fine-tune end to end for 25k steps — [arXiv 2404.00411v3](https://arxiv.org/html/2404.00411v3)

### Inferences
- Every model listed predicts a tendency or residual (explicitly in GraphCast, GenCast, Stormer, ACE2 and Aardvark's processor). Our anomaly-change CNN is already on the consensus path.
- Two models already use a 24 h step (ArchesWeather, Aardvark) and Pangu keeps a 24 h model, so a 1-day step is not a handicap in itself. The real difference is that we use daily means rather than instantaneous states, which smooths fast dynamics and arguably makes the problem easier per step.
- The ensemble/probabilistic generation (GenCast, AIFS-CRPS, FGN, ArchesWeatherGen) exists mainly to fix the blurring caused by MSE rollout training. That blurring is exactly what our day-8 spatial-variance metric penalises.

### Gaps
- The FourCastNet v1 (AFNO) architecture and its patch/step details were not re-fetched. From memory (unverified): a ViT with adaptive Fourier token mixing, 0.25 degrees, 6 h, about 20 variables. FourCastNet 3 (2025) was not researched.
- I could not open the GraphCast supplementary material (PDF too large to fetch). Its residual normalisation details and the 1-degree GraphCast_small spec are from memory only.
- Not covered: NeuralGCM (hybrid dynamical core plus ML, Nature 2024); Keisler 2022 (1-degree GNN, a useful small-scale precedent); FuXi-ENS; regional/LAM models such as Neural-LAM and stretched-grid AIFS, which are the most relevant to our lack of lateral boundary inflow. These should be checked in a regional-models note.

## Q2. Reported quantitative results and compute/data volume

### Takeaway
Global models trained on about 40 years of 6-hourly 0.25-degree ERA5 (roughly 58k samples per year-step, TB-scale) beat IFS HRES on about 90% of targets (GraphCast, Aurora) and ENS on about 97% (GenCast). Compute ranges from about 9-11 V100-days (ArchesWeather at 1.5 degrees) to weeks on 32-192 accelerators. Coarse-resolution, small-compute models (ArchesWeather, Stormer at about 1.4 degrees) remain competitive, which is the most relevant evidence for a small setup.

### Cited Findings
- **GraphCast:** beats the best operational deterministic system on 90% of 1,380 targets; about 4 weeks on 32 TPU v4 — [arXiv](https://arxiv.org/abs/2212.12794); [PDF](https://arxiv.org/pdf/2212.12794)
- **GenCast:** beats ENS on 97.4% of 1,320 targets (99.8% beyond 36 h, CRPS, p<0.05). Tropical-cyclone track relative economic value is 11%, 38% and 106% higher than ENS at 1, 3 and 5 days. Regional wind-power CRPS is about 20% better than ENS up to 2 days. About 8 min per 15-day member on one TPUv5. Member spectra match ERA5; the ensemble mean is blurry — [arXiv 2312.15796v2](https://arxiv.org/html/2312.15796v2). I did not obtain the training-compute figure.
- **WeatherNext 2 / FGN:** beats GenCast on 99.9% of variables and lead times (0-15 days), per Google's blog. The paper says it beats GenCast and ENS across deterministic and probabilistic metrics. I could not extract exact CRPS numbers — [blog](https://blog.google/technology/google-deepmind/weathernext-2/?linkId=27987866); [arXiv 2506.10772](https://arxiv.org/html/2506.10772v1)
- **Pangu-Weather:** 5-day Z500 RMSE of 296.7 vs 333.7 for IFS and 484.5 for FourCastNet — [Nature PDF](https://www.nature.com/articles/s41586-023-06185-3.pdf). Training compute as quoted by Stormer: about 60 days on 192 V100 — [Stormer](https://arxiv.org/html/2312.03876). I did not verify this against the Pangu Methods section; ArchesWeather quotes 2,880 V100-days — [ArchesWeather](https://arxiv.org/html/2405.14527).
- **FuXi:** skilful Z500 lead time (ACC > 0.6) of 10.5 days vs 9.25 for HRES and 9.5 for GraphCast. For T2M, 14.5 days vs 10 for HRES and GraphCast. Comparable to the ECMWF ensemble mean at 15 days — [arXiv 2306.12873](https://arxiv.org/pdf/2306.12873)
- **FengWu:** beats GraphCast on 80% of 880 predictands. 10-day Z500 RMSE of 651 vs 733 m²/s². Skilful Z500 to 10.75 days. About 600 ms per step on an A100 — [arXiv 2304.02948](https://arxiv.org/abs/2304.02948)
- **AIFS (deterministic):** upper-air scores better than IFS by "of the order of 10%" up to 100 hPa (IFS better at 50 hPa). NH Z500 ACC gains more than 12 h. Tropical-cyclone track errors are substantially lower, mainly because the slow propagation-speed bias shrinks, but storms are too weak. Precipitation is mixed: worse than IFS (SEEPS) in the extratropics at short range, better in the tropics. Trained for about 1 week on 64 A100 (40 GB). A 10-day forecast takes about 2.5 min on one A100 — [arXiv 2406.01465](https://arxiv.org/html/2406.01465)
- **AIFS-CRPS:** upper-air CRPS, RMSE and ACC about 5-20% better than IFS ENS at medium range. Precipitation is more skilful than ENS for about the first 8 days, then less (attributed to low spread). Over-dispersive for extratropical Z500, under-dispersive in the tropics. Keeps realistic small-scale variability, unlike MSE-AIFS, whose spectra lose high-wavenumber energy with lead time. Training: about 4 days on 64 H100 (O96) or about 7 days on 128 H100 (N320) — [arXiv 2412.15832](https://arxiv.org/html/2412.15832)
- **Stormer:** trained in under 24 h on 128 A100. Training data is about 190x smaller than Pangu's and 90x smaller than GraphCast's. Days 1-5 are slightly behind GraphCast; from day 6 it beats Pangu and GraphCast, with about 10-20% lower error than GraphCast at day 14. The paper calls it the first 1.40625-degree model to beat IFS, but it is slightly worse than IFS ENS — [arXiv 2312.03876](https://arxiv.org/html/2312.03876)
- **ArchesWeather:** 24 h RMSE for Z500: 48.1 (M), 44.36 (Mx4) vs 42.30 for IFS HRES and 44.31 for Pangu. T850: 0.619 (Mx4) vs 0.625 (HRES). Q700: 0.523 vs 0.556. Compute is about 9-11 V100-days for M (about 2.5 days on 2 A100), vs 2,880 V100-days for Pangu and 7,680 for the NeuralGCM ensemble. The dataset is under 1 TB — [arXiv 2405.14527](https://arxiv.org/html/2405.14527). The paper reports internally inconsistent figures (9 vs 11 V100-days; 300k vs 320k steps).
- **ArchesWeatherGen:** about 45 V100-days of training and about 1 min per 15-day member on an A100. Claims to beat IFS ENS and NeuralGCM on all WeatherBench headline variables except NeuralGCM's geopotential — [arXiv 2412.12971](https://arxiv.org/abs/2412.12971)
- **ACE2:** stable 81-year rollouts and a 1,000-year run without drift. Realistic ENSO precipitation response and MJO. Tropical-cyclone frequency is overpredicted by about 28%. Weather skill is below IFS and GraphCast. A 4-degree version performs nearly as well as the 1-degree one. Training takes about 4.5 days on 8 H100 per model; runtime is about 1,500 simulated years per day on one H100 — [arXiv 2411.11268](https://arxiv.org/html/2411.11268)
- **Aurora:** beats IFS HRES on 92% of targets at 0.1 degrees (RMSE reductions up to 24%), and beats IFS and GraphCast on more than 91% at 0.25 degrees. Pre-trained on more than 1M hours of data; 150k steps on 32 A100 (about 2.5 weeks) — [arXiv 2405.13063](https://arxiv.org/html/2405.13063)
- **Aardvark:** beats GFS on the headline variables. It approaches HRES for T2m and MSLP at long leads, beats HRES on 10 m wind from day 7 and on T850 from day 8, but is worse on Z500 to day 8. Station forecasts are skilful to about 10 days and beat NDFD for wind over CONUS. Uses about 8% of the observations NWP uses. Inference takes about 1 s on 4 A100 — [arXiv 2404.00411v3](https://arxiv.org/html/2404.00411v3)

### Inferences
- The "small compute" models (ArchesWeather about 10 V100-days, Stormer under 1 day on 128 A100, ACE2 about 36 H100-days) still use about 40 years of global 6-hourly data, so millions of grid-column samples. Our roughly 4,700 x 1,152 cells of data is 2-3 orders of magnitude smaller, so architectural capacity has to be kept small and the inductive bias local.
- AIFS's finding that rollout training reduced the tropical-cyclone slow-propagation bias is directly relevant to our "cyclones do not move" weakness. The flip side is weaker intensity, the same blurring trade-off.

### Gaps
- I did not obtain GenCast training compute, the FGN CRPS table, the Prithvi WxC quantitative results, or Aurora's small-model scores.
- WeatherBench 2 leaderboard numbers were not fetched, so I make no cross-model table at a common resolution.

## Q3. Which ingredients matter most in ablations

### Takeaway
The ablation evidence ranks roughly as follows:
1. Predicting tendencies, not states.
2. Multi-step rollout fine-tuning, which matters most at long leads.
3. A probabilistic/CRPS loss or a generative head to stop blurring.
4. Loss weighting toward the surface and the variables that are scored.
5. Seed ensembling and multi-interval averaging.
6. Pre-training on more or more diverse data.
7. Model size, with log-scale gains of about 6% per 10x.

Backbone innovations (column attention, multimesh) also help, but they are tied to 3D global data.

### Cited Findings
- **Stormer ablations:** randomised multi-interval training beats single-interval at all leads, with the gap growing with lead time. Forecasting dynamics beats forecasting absolute states. Pressure-weighted loss helps. Multi-step fine-tuning "significantly improves long lead times". Per-variable embedding beats patch embedding; adaLN beats additive lead-time embedding. Larger models and smaller patches help consistently — [arXiv 2312.03876](https://arxiv.org/html/2312.03876)
- **ArchesWeather ablations (24 h Z500):**
  - S-size: Pangu-S 66.7; ArchesWeather-S 49.3, or 55.1 without Cross-Level Attention.
  - M-size: Pangu-M 58.7; ArchesWeather-M 48.0, or 51.8 without Cross-Level Attention.
  - Removing multi-step fine-tuning changes 48.0 to 48.7, a small effect at 24 h.
  - The 4-seed ensemble improves Z500 from 48.1 to 44.36 (about 8%).
  - [arXiv 2405.14527](https://arxiv.org/html/2405.14527)
- **FuXi:** a single model without the cascade degrades significantly at long leads because of accumulated error — [arXiv 2306.12873](https://arxiv.org/pdf/2306.12873)
- **AIFS-CRPS vs MSE-AIFS:** MSE training loses small-scale energy with lead time; CRPS training does not. Without truncation of the reference field during rollout, fields gained spurious small-scale energy — [arXiv 2412.15832](https://arxiv.org/html/2412.15832)
- **GenCast:** individual members keep ERA5-like spectra, while the ensemble mean is blurry — [arXiv 2312.15796v2](https://arxiv.org/html/2312.15796v2)
- **Aurora:** pre-training vs training from scratch gives about 25% better results at 0.1 degrees. More diverse pre-training data systematically helps, especially for extremes. About 6% improvement per 10x model size — [arXiv 2405.13063](https://arxiv.org/html/2405.13063)
- **ACE2:** hard conservation and non-negativity constraints plus a 2-step loss give century-scale stability. 4 degrees is nearly as good as 1 degree for climate statistics — [arXiv 2411.11268](https://arxiv.org/html/2411.11268)
- **ArchesWeather:** a final training stretch on recent years (2007-2018) reduces distribution-shift error. Bilinear upsampling plus a convolutional head reduces checkerboard artefacts — [arXiv 2405.14527](https://arxiv.org/html/2405.14527)
- **Aardvark:** fine-tuning the processor on its own (encoder-produced) inputs and then end-to-end fine-tuning give about 6% MAE reduction for temperature and 1-3% for wind — [arXiv 2404.00411v3](https://arxiv.org/html/2404.00411v3)

### Inferences
- The effect of rollout fine-tuning depends on lead time: it is negligible at 24 h (ArchesWeather) but large at 7-14 days (Stormer, FuXi). Our grading at days 4 and 8 is in the regime where it matters.
- Seed ensembling is the cheapest reliable RMSE gain (about 8% at 24 h in ArchesWeather). It reduces spatial variance, though, which hurts our day-8 anti-blur metric. Expect a measurable conflict between ensemble averaging and the variance score.

### Gaps
- I found no clean ablation isolating "noise added to inputs during training" in the papers I fetched, apart from AIFS-CRPS and GenCast noise as the ensemble mechanism. GraphCast and other papers may contain such ablations, but I did not verify them.
- I found no ablation in these papers of daily-mean (time-averaged) targets versus instantaneous ones.

## Q4. What transfers to a tiny regional, daily-mean, CPU-only, ~4.7k-sample setting, and what conflicts with locality/causality tests

### Takeaway
Transfer the **training recipe**, not the architectures:
- normalised residual targets
- curriculum rollout fine-tuning to about 8-10 days
- per-variable and area loss weighting tuned to the graded metrics
- a CRPS-style or variance-preserving loss against blur
- adaLN-style conditioning on season
- a late fine-tune on the most recent years
- hard non-negativity and moisture-budget constraints on rain

Do not adopt global spectral mixing (SFNO/AFNO), global or shifted-window attention, or multimesh long-range edges. They violate the "shock spreads at most 8 cells/day" test and are over-parameterised for 4.7k samples. Stochastic generators are only acceptable if run deterministically (fixed seed), so that a zero shock gives identical output.

### Cited Findings (evidence base for the transfer judgements)
- GraphCast's multimesh has coarse-level edges that span large distances, so one processor step can propagate information far beyond 8 cells — [GraphCast PDF](https://arxiv.org/pdf/2212.12794); [summary](https://gist.github.com/wolfram77/fffffc414960fc9d3dab3bddd9163a8a)
- SFNO mixes in spherical-harmonic (global) space — [arXiv 2306.03838](https://arxiv.org/abs/2306.03838). Stormer uses full ViT self-attention over all tokens — [arXiv 2312.03876](https://arxiv.org/html/2312.03876)
- AIFS uses sliding or shifted-window attention along latitude bands, which is local per layer but compounds over 16 layers — [arXiv 2406.01465](https://arxiv.org/html/2406.01465)
- Aardvark and ArchesWeather show that 24 h steps on a 1.4-1.5-degree grid work — [Aardvark](https://arxiv.org/html/2404.00411v3); [ArchesWeather](https://arxiv.org/html/2405.14527)
- ACE2 enforces non-negativity of precipitation and moisture and conserves column moisture through a physics corrector — [arXiv 2411.11268](https://arxiv.org/html/2411.11268)
- Rollout training reduced the tropical-cyclone slow-propagation bias in AIFS — [arXiv 2406.01465](https://arxiv.org/html/2406.01465)
- AIFS-CRPS and FGN train ensembles directly with (almost) fair CRPS — [AIFS-CRPS](https://arxiv.org/html/2412.15832); [FGN](https://arxiv.org/html/2506.10772v1)

### Inferences: likely to transfer (ranked by expected value for our metrics)
1. **Curriculum rollout fine-tuning to 8-10 steps with backprop through time** (GraphCast, AIFS, Stormer). It is cheap for a 239k-parameter CNN on a 32x36 grid. It targets the day-4 and day-8 RMSE and should help cyclone propagation (AIFS evidence). Combine it with a pushforward or replay-buffer variant (Aurora, FengWu) if memory or instability becomes a problem.
2. **A loss that does not reward blur.** Pure MSE rollouts will shrink day-8 variance (AIFS-CRPS spectra, GenCast ensemble mean). Options:
   - a small (2-4 member) CRPS/afCRPS-trained noise-conditioned head, run with a fixed seed at inference to stay deterministic
   - a spatial-variance or spectral penalty added to MSE
   - for rain, keeping the two-part head (occurrence and amount) trained on CSI-relevant thresholds
3. **Normalise residual targets by the std of daily tendencies per variable** (GraphCast, GenCast), not by state anomalies alone. Then set **per-variable loss weights** matched to the grading, analogous to the GraphCast/Stormer surface weights (T2m 1, winds and MSLP 0.1), which are clearly tuned to their own metrics.
4. **Season / day-of-year conditioning through adaptive LayerNorm or FiLM** (ArchesWeather, Stormer). It is cheap and helps monsoon-onset regime changes.
5. **Fine-tune late on the most recent training years** (ArchesWeather's 2007-2018 trick) to reduce shift if the test years are after 2017.
6. **Physics-style hard constraints for rain** (ACE2): non-negativity via the head (already present) plus a soft column-moisture budget, rain ≈ −ΔTCWV + convergence of moisture flux from u/v and TCWV. This directly addresses weak wind-to-rain coupling, because it gives the network an explicit pathway from wind convergence to rain.
7. **Cross-variable per-cell mixing** as a cheap analogue of ArchesWeather's column attention: 1x1 conv / channel-attention blocks between spatial convs, letting wind, TCWV and MSLP interact at each cell without widening the spatial footprint.
8. **Seed ensembles (2-4 CNNs)** fit the CPU budget, since each rollout is milliseconds at 239k parameters. The averaging blurs, though, so either evaluate the variance cost or use the members to feed a CRPS-style spread rather than averaging.
9. **Multi-interval training** (Stormer's random dt in {1, 2} days, averaging combinations at inference) is a plausible cheap gain at long leads. It is untested at daily-mean resolution.

### Inferences: unlikely to transfer, or conflicting
- **Global spectral or attention mixing** (SFNO/FourCastNet, Stormer ViT, Prithvi global attention), **multimesh GNN long edges** (GraphCast/GenCast) and **deep Swin U-Nets with downsampling** (Pangu, Aurora, ArchesWeather) all have receptive fields covering most of a 32x36 domain in one step. They would likely fail "shock stays near its box on day 1 / spreads at most 8 cells/day". The current 7-layer 3x3 CNN has a receptive radius of about 7 cells per step, which matches the rule by construction. Any added layers or dilations must keep the per-step radius at 8 cells or less.
- **Diffusion or flow-matching generators** (GenCast, ArchesWeatherGen): 20+ denoiser evaluations per step x 10 steps is probably affordable on CPU for a tiny net. They are stochastic, though, which breaks "zero shock = identical output" unless the noise is seeded deterministically, and they need far more data to learn distributions. Not recommended at 4.7k samples.
- **Foundation-scale pre-training** (Aurora 1.3B, Prithvi 2.3B): its gains come from more than 1M hours of data, which we cannot use, and the parameter counts are 4-5 orders of magnitude over our budget.
- **Cascades of lead-time-specific models** (FuXi) and **Pangu's hierarchy of step models**: possible in principle (for example separate fine-tuned weights for days 1-4 and 5-10), but each model sees fewer effective samples. Treat as low priority.
- **Lateral boundary inflow:** global models have no boundaries, so none of them offers a solution. The regional models (Neural-LAM, stretched-grid AIFS) are the right references; see Gaps.

### Gaps
- No source here tests any of these ingredients at about 5k samples or on daily-mean targets. All transfer claims are inferences.
- I have not verified how the grading supplies boundary information. Whether future boundary forcing is allowed determines whether LAM-style boundary nudging is even possible.

## Q5. Are any published weights small enough or adaptable for fine-tuning or distillation onto our grid?

### Takeaway
Several global checkpoints are openly licensed:
- AIFS Single 1.0: CC BY 4.0
- GraphCast/GenCast: code Apache 2.0, "other materials" CC BY 4.0 per the current README
- ACE2-ERA5 and Prithvi WxC: on Hugging Face
- Aurora small: about 500 MB download

None of them can run directly from our inputs. All need full 3D atmospheric initial states at 6 h resolution and are 10-10,000x larger than our model. Their realistic use is as an **offline teacher** generating extra synthetic training pairs, and only if competition rules allow non-ERA5-2005-2017 data and models. They cannot be shipped in the 5 s CPU runtime.

### Cited Findings
- **AIFS Single 1.0:** weights CC BY 4.0, code Apache 2.0. About 31 km grid, 13 pressure levels, 6 h step. Its inputs and outputs include T2m, 10 m u/v, MSLP and total column water; total precipitation is output only — [Hugging Face ecmwf/aifs-single-1.0](https://huggingface.co/ecmwf/aifs-single-1.0)
- **GraphCast/GenCast repo:** code Apache 2.0, "All other materials ... CC BY 4.0". The README now labels them legacy "WeatherNext Graph / Gen" — [GitHub google-deepmind/graphcast](https://github.com/google-deepmind/graphcast). My memory says earlier READMEs listed CC BY-NC-SA 4.0 for the weights and offered a 1-degree "GraphCast_small" checkpoint; I could not confirm either on the current page, so check the docs before relying on it.
- **ACE2-ERA5:** checkpoint on Hugging Face (allenai/ACE2-ERA5), code at github.com/ai2cm/ace. 1 degree, 6 h, about 450M parameters. The paper page states no licence for the weights — [arXiv 2411.11268](https://arxiv.org/html/2411.11268)
- **Aurora:** `AuroraSmallPretrained` checkpoint (about 500 MB download) plus specialised weather, air-quality and wave models. A `finetuning` directory exists. The repo links LICENSE.txt but the page did not state its terms; commercial use requires emailing Microsoft — [GitHub microsoft/aurora](https://github.com/microsoft/aurora)
- **Prithvi WxC:** pre-trained model and fine-tuning workflows released openly on Hugging Face — [arXiv 2409.13598](https://arxiv.org/abs/2409.13598)
- **ArchesWeather/Gen:** code, models and the full pipeline are open source — [arXiv 2412.12971](https://arxiv.org/abs/2412.12971)
- **WeatherNext 2:** available as a Google data/API product. A Hugging Face "WeatherNext2 FGN" page is an unofficial reproduction, not Google's model — [HF OneScience-Group/WeatherNext2](https://huggingface.co/OneScience-Group/WeatherNext2); [Google developers](https://developers.google.com/weathernext/guides/models-wn2)

### Inferences
- **Direct fine-tuning onto our grid is impractical.** Their input contracts (multi-level winds, temperature, humidity and geopotential at 6 h) do not match our 6 daily-mean surface fields. Their parameter counts (about 36.7M for GraphCast, 229M or more for the AIFS processor, about 450M for ACE2, 1.3B for Aurora) cannot meet 4-core, under-5-s inference over 10 steps. They are also global, so sub-setting India would still need global context.
- **Distillation is the only plausible route.** Run AIFS (CC BY 4.0, includes all 6 of our variable types) or ArchesWeather offline from ERA5 initial conditions, average to daily means over the India box, and use the outputs as extra (input, 1-10-day target) pairs or as soft targets for rollout fine-tuning. This could multiply our effective sample count and teach cyclone propagation and wind-rain coupling. Caveats:
  - The teacher's own blur would be inherited; use AIFS-ENS/CRPS members rather than ensemble means.
  - It needs GPU time.
  - It may breach a training-data restriction if one exists.
- AIFS's licence (CC BY 4.0) and variable overlap make it the best teacher candidate. ArchesWeather (1.5 degrees, 24 h step, small compute) is the cheapest to run but is closer to a global upper-air model, and I did not verify its surface variable set beyond "4 surface variables".

### Gaps
- Not verified: exact licence terms for the Aurora weights, ACE2 weights, ArchesWeather weights and the GraphCast_small checkpoint; the Aurora small parameter count; whether AIFS can run on CPU in reasonable time.
- Unknown whether the competition permits external pre-trained models or teacher-generated data. This should be checked against the task rules before investing in distillation.
