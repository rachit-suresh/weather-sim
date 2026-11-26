# Physics-informed / hybrid ML weather models and India / monsoon-specific ML forecasting (2022-2026)

Context for the report writer: our system is a 32x36, 1-degree daily-mean India emulator with 6 surface variables (t2m, rain, u10, v10, msl, tcwv). It runs autoregressively for 10 days on CPU in under 5 s and is deterministic. The best model is a local residual CNN in anomaly space. A physics hybrid does semi-Lagrangian advection of the anomaly with a learned displacement (capped at 4 cells/day, bilinear grid_sample) and then a source/sink CNN. The hybrid tracked Amphan but scored about 2-3 points lower, and blending the two models broke shock consistency. In the Arabian Sea wind -40% shock, rain barely changes, and the response fails to "spread outward" in 3 of 5 windows.

Research scope note: about 17 search/fetch calls. Many items are arXiv preprints from 2025-2026 and are not peer-reviewed. Most fetches returned abstracts only, so numbers are limited to what those pages showed. Anything taken from background knowledge rather than a fetched page is marked **[unverified this session]**.

---

## Q1. Hybrid physics-ML designs: which ideas improve physical consistency, transport and causality, and which could make our physics hybrid as accurate as the CNN?

### Takeaway
The most directly transferable recent result is PARADIS (2026). It advects a *learned latent subspace* with a neural semi-Lagrangian operator inside a Lie-Trotter advection, diffusion, reaction split, with a learned per-channel blend weight. It matches GraphCast at 72 h and beats it at 6 h, at about 1/10 or less of the usual training cost. This points to a specific fix for our hybrid: advect learned features instead of raw anomalies, and do the CNN correction *inside* each step rather than blending two models' outputs afterward. NeuralGCM, a differentiable dynamical core with learned column physics, gives the best overall evidence that hybrids can match pure ML. Its scale (GPU/TPU, global, 3-D) does not fit our CPU 5 s budget, but its ideas do.

### Cited Findings
**NeuralGCM (Kochkov et al., Nature 2024)**
- The model pairs a differentiable dynamical core with learned physics. It is competitive with ML models for 1-10 day forecasts and with the ECMWF ensemble for 1-15 days. Google reports that the NeuralGCM ensemble beats ECMWF-ENS 95% of the time for 2-15 day forecasts — [Google Research pub](https://research.google/pubs/neural-general-circulation-models-for-weather-and-climate/); [Google blog](https://research.google/blog/fast-accurate-climate-modeling-with-neuralgcm/); [arXiv 2311.07222](https://arxiv.org/abs/2311.07222v3)
- Climate: the 2.8-degree deterministic model's 1980-2020 temperature error is one-third that of CMIP AMIP models. At 140 km it shows realistic tropical-cyclone frequency and tracks — [Google Research pub](https://research.google/pubs/neural-general-circulation-models-for-weather-and-climate/)
- Cost: the authors claim orders-of-magnitude savings over conventional GCMs and say it "can run on a laptop". A Nature referee noted 130,000 TPU-hours of training cost at 0.7 degrees — [Nature peer review file](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-024-07744-y/MediaObjects/41586_2024_7744_MOESM2_ESM.pdf). Code and weights are open for non-commercial use — [Google blog](https://research.google/blog/fast-accurate-climate-modeling-with-neuralgcm/)
- It does not extrapolate to substantially different future climates — [Google blog](https://research.google/blog/fast-accurate-climate-modeling-with-neuralgcm/)
- **NeuralGCM-precip (Yuval, Langmore, Kochkov, Hoyer; arXiv Dec 2024; Science Advances 2026, doi 10.1126/sciadv.adv6891).** The original NeuralGCM did not explicitly represent P and E. This version adds them, at 2.8 degrees. It trains atmosphere and evaporation on ERA5 and precipitation on IMERG V07 (2001-2018). Reported results: lower biases, a more realistic precipitation distribution, better extremes, a better diurnal cycle, and better mid-range precipitation forecasts than the ECMWF ensemble — [arXiv 2412.11973](https://arxiv.org/abs/2412.11973?context=cs); [Google blog](https://research.google/blog/neuralgcm-harnesses-ai-to-better-simulate-long-range-global-precipitation/)

**PARADIS, "Learning to Advect: A Neural Semi-Lagrangian Architecture for Weather Forecasting" (Pereira et al., ECCC/UBC/Saskatchewan; arXiv 2601.21151 v2, May 2026, preprint)** — [arXiv 2601.21151v2](https://arxiv.org/html/2601.21151v2)
- A depthwise-separable conv predicts transport velocities from the latent state. A pointwise down-projection selects a *low-dimensional subspace* of latent channels to advect. The model backtracks departure points and samples with **bicubic `grid_sample`**, then lifts the result back to the full latent space. The model also learns *which* features to transport.
- Each substep runs a first-order **Lie-Trotter splitting**: (1) advection, blended with a learned per-channel weight alpha; (2) diffusion, as depthwise-separable mixing on a grid coarsened 4x and added back as a residual; (3) reaction, as two pointwise channel MLPs with SiLU conditioned on topography and land-sea mask. The model predicts 6 h increments from the current and previous states.
- Stability is "not CFL-limited". The limit is a Lipschitz condition on the velocity field. The fetched content did not give substep counts or displacement caps.
- Toy ablation: a 3-layer U-Net with about 44k parameters cannot move a Gaussian tracer beyond its receptive field. One NSL layer with 20 parameters relocates it accurately. The authors say ablations show the SL component is central to spectral and activity behaviour, but the ablation tables were not in the fetched text.
- z500 RMSE for 2020 (6 h / 72 h / 240 h): PARADIS 10.8 / 124 / 806; GraphCast 14.7 / 124 / 732; HRES 25.7 / 137 / 803. PARADIS trails GraphCast, FuXi and Baguan at 240 h, and the authors say it is "not uniformly RMSE-optimal". It keeps spectral amplitude and activity better than GraphCast, FuXi and Baguan, which is relevant to anti-blur.
- Cost: about 0.66 GPU-years on 32 H100s. A 10-day 0.25-degree forecast takes about 35 s. Training was a 1-degree pretrain then a 0.25-degree transfer, followed by autoregressive fine-tuning from 12 h to 72 h with backprop through 2 steps. The loss is a "pseudo-reversed Huber" (quadratic for large errors, linear for small ones).
- Limitations: deterministic; **no precipitation or diagnostic variables**; no conservation diagnostics reported.

**ClimODE (Verma, Heinonen, Garg; ICLR 2024 Oral)** — [arXiv 2404.10024](https://arxiv.org/abs/2404.10024v1); [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/21f3c05fc5afcaebd8a71d7e98a1b171-Abstract-Conference.html)
- Weather is modelled as *advection*, solving the **continuity equation**, which has a transport term and a *compression (divergence) term*, as a neural ODE over the globe. The flow field comes from a small network with global attention plus local convolutions. Sources are handled by a probabilistic emission model.
- The authors claim state of the art among data-driven methods in global and regional forecasting at 5.625 degrees, with an order of magnitude fewer parameters (about 2.8M). Exact benchmark numbers were not verified this session.

**PDE-Refiner (Lippe et al., NeurIPS 2023)** — [arXiv 2308.05732](https://arxiv.org/abs/2308.05732); [NeurIPS slides](https://neurips.cc/media/neurips-2023/Slides/71688.pdf)
- The authors identify neglect of non-dominant, often high, spatial frequencies as the main cause of rollout failure. The fix is a diffusion-style multistep refinement. Tested on 1D Kuramoto-Sivashinsky and 2D Kolmogorov flow, it beats neural, numerical and hybrid baselines, with "up to 30% longer" accurate rollouts according to the slides. The denoising objective acts as spectral data augmentation.

**Conservation layers and constraints**
- **ACE2 (Ai2, arXiv 2411.11268)** enforces *exact* global dry-air-mass and moisture conservation through a corrector. It does not conserve energy — [arXiv 2411.11268](https://arxiv.org/html/2411.11268v1)
- A pilot study of ACE2's conservation and non-negativity corrector found that it cut negative-precipitation frequency from 33.1% to 0%. However, **precipitation RMSE was 2.46% higher over 10 days** (1.75% over 30 days) with correction at both stages. This was a small pilot with 3 seeds and 7 starts — [arXiv 2610.23561](https://arxiv.org/pdf/2610.23561)
- In a fast-radiative-response study with ML emulators, closing the moisture budget by rescaling *evaporation* gave a response with the **wrong sign**. Budget closure alone did not guarantee correct causal physics — [arXiv 2602.16090](https://arxiv.org/pdf/2602.16090) (snippet-level only, not fetched)
- **CondensNet / PCNN-GCM (npj Clim. Atmos. Sci. 2025)** uses a base net plus a correction net with an adaptive physical constraint on condensation, which keeps hybrid GCM runs stable over the long term — [Nature npj](https://www.nature.com/articles/s41612-025-01269-5)
- A temperature and humidity post-processing study (arXiv 2212.04487) compares enforcing analytic constraints in the architecture with enforcing them in the loss — [arXiv 2212.04487](https://arxiv.org/pdf/2212.04487)
- **[unverified this session]** Beucler et al. (PRL 2021), "Enforcing analytic constraints in neural networks emulating physical systems", is the classic source for hard constraints via a fixed output layer versus soft constraints via a loss penalty. Hard constraints satisfied conservation to machine precision with little accuracy loss in their setup.

### Inferences (applicability to our system)
- **Why our hybrid loses 2-3 points:** it advects the *raw anomaly* of each variable and corrects afterward. PARADIS (a) advects a *learned latent subspace*, so the model can choose features for which advection is a good operator, (b) blends advection with a learned per-channel alpha, and (c) runs diffusion and reaction *inside every substep*. This lets the model fall back to CNN-like local updates where transport is wrong. Suggested change: encoder (1x1 conv to about 16-32 channels), then K substeps of [NSL advection of a subset of channels with alpha-blend, then a coarse diffusion conv, then a pointwise reaction MLP], then a decoder. This keeps transport *structural*, which preserves Amphan tracking and locality, without discarding CNN accuracy. It is one model, so no output blending is needed, and that blending is what broke shock consistency for us.
- **Substeps instead of a 4-cell cap:** the NSL limit is Lipschitz-smoothness of the velocity field, not CFL. Splitting the daily step into 2-4 substeps, each displacing at most 1-2 cells, should reduce interpolation smearing and keep the velocity field smooth. Bicubic rather than bilinear sampling reduces numerical diffusion, which matters for anti-blur, though bicubic can overshoot, so clamp rain at zero.
- **"Spread outward" failures:** a pure SL step only *moves* anomalies. ClimODE's continuity equation adds the *compression term* (-u * div V). Diffusion-type spreading needs an explicit diffusion block, as in PARADIS. Our hybrid has neither, so a localized Bay of Bengal anomaly is displaced but not spread. A learned, non-negative-diffusivity Laplacian step per substep (one 3x3 conv with a fixed Laplacian stencil times a learned positive coefficient field) is cheap, deterministic and local.
- **Cost fit:** on a 32x36 grid with K=4 substeps, about 20 channels, 10 days and `grid_sample` on CPU, the runtime should be well under 1 s. `grid_sample` forward is deterministic on CPU. NeuralGCM itself (global spectral core, JAX) does not fit our budget or variables, so we should borrow only its *pattern* (fixed physics core plus learned tendencies, trained end-to-end through multi-step rollouts).
- **PDE-Refiner:** it targets exactly the anti-blur / high-frequency problem. It needs several denoising passes per step and injected noise, so it is only compatible with determinism if the noise seed is fixed. Its cost would be 3-8 times the model per step, which is likely still under 5 s at 32x36. It is probably lower priority than spectral or anti-blur loss terms, but it gives evidence that the loss of high frequencies is what drives rollout degradation.
- **Hard conservation correctors can cost rain RMSE** (ACE2 pilot +2.5%) and can produce wrong-sign responses if the wrong term is rescaled. Prefer a structural budget pathway plus a soft penalty, not a post-hoc projection (see Q2).

### Gaps
- PARADIS ablation numbers (advection off versus on) and substep counts were not retrieved; the appendix was beyond the fetch window.
- ClimODE's exact RMSE numbers were not retrieved.
- No paper found that tests a semi-Lagrangian hybrid on *daily-mean, regional, 1-degree* data or on causal shock tests.
- Flux-form or Hamiltonian networks for weather: no 2022-2026 weather-scale evidence found in this session.
- NeuralGCM inference speed figures (for example simulated days per hour on one device) were not verified this session.

---

## Q2. Coupling moisture transport and rain physically (column water budget) in a learned model

### Takeaway
Direct evidence is thin but points one way. Moisture-flux-convergence (MFC) features carry the strongest predictive signal for rain among budget terms. Removing them measurably lowers CSI (0.324 to 0.296 in one 2026 nowcasting preprint). NeuralGCM only got good precipitation after P and E were made *explicit* model outputs trained against observations. Hard budget projections help physical plausibility but can slightly raise rain RMSE or flip response signs if misapplied. For our wind-shock failure, the most likely cause is that rain has no structural dependence on wind-driven moisture convergence.

### Cited Findings
- **MZ-Rain (arXiv 2609.04864, preprint)** builds moisture-budget terms into a station precipitation nowcaster. **Removing the convergence term dropped CSI from 0.3243 to 0.2961** and raised the false-alarm ratio. MFC had the strongest association with future rain among budget components (Spearman 0.345-0.368) — [arXiv 2609.04864](https://arxiv.org/pdf/2609.04864) (search snippet; full text not fetched)
- A daily-rain LSTM/Transformer study (Indonesia, JMG BMKG) found vertically integrated moisture flux divergence (VIMD) was the **most important predictor** for daily rain in convective and monsoonal regions — [JMG BMKG](https://jmg.bmkg.go.id/jmg/index.php/jmg/article/download/1194/511). Note: divergence and convergence are the same quantity with opposite sign, so this does not contradict MZ-Rain.
- NeuralGCM-precip: the authors modified NeuralGCM to represent P and E explicitly, training P on IMERG and E on ERA5. This improved bias, extremes and the diurnal cycle — [arXiv 2412.11973](https://arxiv.org/abs/2412.11973?context=cs)
- ACE2 enforces exact global moisture conservation, with P adjusted so the global budget closes — [arXiv 2411.11268](https://arxiv.org/html/2411.11268v1). The corrector raised 10-day P RMSE by about 2.5% in a pilot — [arXiv 2610.23561](https://arxiv.org/pdf/2610.23561). Closing the budget via E gave a wrong-sign fast response — [arXiv 2602.16090](https://arxiv.org/pdf/2602.16090)
- PID-GAN (precipitation nowcasting) injects the moisture conservation equation as a physics loss — [arXiv 2406.10108](https://arxiv.org/pdf/2406.10108)
- A CNN post-processor of GFS integrated vapour transport (IVT) improved IVT skill out to 7 days, per a search summary of an AGU 2019 study. It is older and about IVT, not rain — reference only.

### Inferences (applicability to our system)
- **Proposed budget pathway, one daily step:**
  - Equation: tcwv(t+1) = tcwv(t) - dt * div(q_eff * V_eff) + E_hat - P_hat.
  - V_eff is a *learned effective steering wind*. It comes from a small CNN of (u10, v10, msl gradient, tcwv) or is a learned linear map of (u10, v10). This is needed because moisture transport is dominated by about 850 hPa flow, which the 10 m winds only proxy.
  - q_eff * V_eff is approximated as tcwv * V_eff, in flux form with a centered or upwind finite-volume divergence.
  - P_hat = softplus(f_theta(MFC, tcwv, tcwv anomaly, t2m, ...)), with MFC passed as an **explicit input channel**. E_hat is a small learned term.
  - The loss includes the ERA5 rain target plus a *soft* budget-residual penalty. Do not hard-project.
- **Why this helps the Arabian Sea wind -40% shock:** with MFC as an explicit, structural input to rain, weakening the onshore westerlies reduces convergence at the Western Ghats and lowers rain *by construction*. Today the CNN can ignore the wind channels for rain. A flux-form transport also makes the warm-and-moist Bay of Bengal shock propagate along the flow, which is causal spreading.
- **Monotonicity checks:** a regression test that the derivative of rain with respect to MFC is positive (for example with a finite-difference shock on V_eff) is cheap and deterministic. MZ-Rain's ablation suggests such a feature should also help heavy-rain CSI.
- **ERA5 caveat:** daily-mean ERA5 tcwv, P and E do not close the budget exactly, because of analysis increments and because daily-mean winds times daily-mean q is not the same as the daily mean of the product. A soft penalty with a learned residual term is safer than an exact constraint. Both the ACE2 pilot and 2602.16090 warn that exact projection can hurt RMSE or response sign.

### Gaps
- I found no peer-reviewed study showing that a budget-constrained ML model improves *wind-shock rain response* specifically. The causal-response benefit is our inference.
- No India-specific ML study found that uses MFC as an explicit structural term in a daily forecast emulator.
- No numbers found on how large the ERA5 daily-mean budget residual is over India.

---

## Q3. India / South Asian monsoon-specific ML work (daily to 10-day, plus onset, intraseasonal, cyclones, extremes)

### Takeaway
Global AI models (GraphCast, AIFS, GenCast, FuXi, Pangu, FourCastNet, Aurora) are skilful for large-scale monsoon fields. All of them **under-predict extreme rain and suppress variance**: GraphCast keeps about 63% of temporal variance and shows a slight dry bias over the core monsoon zone. Errors against station, gauge and satellite observations are **15-45% larger** than errors against reanalysis. In India, the biggest operational ML success is *onset* forecasting (NeuralGCM + AIFS + IMD climatology blend, sent to 38M farmers in 2025). Deep-learning MISO prediction is skilful to about 18 days (IIT Delhi). Bay of Bengal cyclone ML is mostly small LSTM/CNN track studies without operational benchmarks.

### Cited Findings
**Evaluations of global AI models over India / South Asia**
- **"MAUSAM" (Gupta, Sheshadri, Suri; arXiv 2509.01879, Sep 2025)** evaluated FourCastNet, SFNO, Pangu-Weather, GraphCast, Aurora, AIFS and GenCast against stations, rain gauges and geostationary satellites. Results: **errors against observations are 15-45% larger than against reanalysis**; strong skill for large-scale temperature, winds, precipitation and cloud cover; *underprediction of extreme precipitation*, divergent cyclone tracks and mesoscale KE-spectrum errors; AIFS is most consistent, with GraphCast and GenCast strong — [arXiv 2509.01879](https://arxiv.org/abs/2509.01879)
- **GraphCast ISM study (Luitel, Mukhopadhyay [IITM], Manmeet Singh, Juneja, Dhanuka; arXiv 2607.11905 v3, Oct 2026)**, JJAS 2021-2024, at 24/48/72 h against ERA5 and IMERG. Core-zone rain bias is **-0.7 to -1.2 mm/day** (the authors call this "near zero"; it is a slight dry bias). Daily correlation with IMERG is **0.89-0.92**. The **spectral variance ratio is about 0.63**. The 95th percentile and extremes are under-represented. Q1/Q2 heating and moistening structure is captured, but northward-propagating intraseasonal variability degrades — [arXiv 2607.11905](https://arxiv.org/abs/2607.11905). **Conflict:** a search snippet of an earlier version described a *wet* bias and "strong suppression" of variance. v3 explicitly says it is "not a wet bias or variance collapse", so use v3.
- 2023-season comparisons (RMetS ECR conference poster, conference-level only): FuXi has a systematic dry bias but the lowest RMSE. GraphCast has the best SEEPS at days 1, 5 and 9. IFS has the best SAL structure. Pangu has no native precipitation, so a proxy was used. AIFS was judged best overall in one study — [RMetS poster](https://www.rmets.org/conference/early-careers-and-student-conference-2025/poster-presentations/assessing-skill-global-ai); [EGU26-7223](https://meetingorganizer.copernicus.org/EGU26/EGU26-7223.html). These are conference abstracts and the numbers were not verified.

**Monsoon onset (operational ML use)**
- In 2025, UChicago's Human-Centered Weather Forecasts Initiative, Google NeuralGCM and India's Ministry of Agriculture **blended NeuralGCM, ECMWF AIFS and IMD historical rainfall**. They sent weekly regional onset forecasts by SMS to about **38 million farmers in 13 states**. The forecasts reportedly caught the about 20-day stall of the 2025 monsoon after early June arrival, which "no other available forecast" predicted (institutional claim) — [Google blog](https://blog.google/technology/research/indian-farmers-monsoon-prediction/); [UChicago](https://climate.uchicago.edu/news/artificial-intelligence-is-helping-indian-farmers-adapt-to-climate-change-forecast-accurately-predicting-an-unusual-monsoon-season-reached-38-million/); [Berkeley News](https://news.berkeley.edu/2025/09/16/ai-revolutionizes-weather-prediction-to-help-farmers-in-india/)
- A benchmark paper (Masiwal, ..., Boos, Hassanzadeh; arXiv 2602.03767, Feb 2026) finds that AI weather models skilfully forecast a regional agricultural onset index "weeks in advance", out-of-sample, using a decision-oriented framework. The abstract does not name the models or give numbers — [arXiv 2602.03767](https://arxiv.org/abs/2602.03767)

**Intraseasonal (MISO / BSISO)**
- **Transformer MISO-index prediction (Anirudh, Raj, Sandeep, Kodamana, Sabeerali; IIT Delhi; JGR: Machine Learning and Computation 2025, vol. 2(2))** is skilful to **18 days** for 2018-2022 and outperforms operational NWP. A conference abstract reports phase skill to 21 days and amplitude skill to 14 days, which may be a different metric — [library record](https://library.kab.ac.ug/Record/doaj-art-103c51d2cce3467db171666796333392/Description); [WMO-IWM8 poster](https://wmo-iwm8.tropmet.res.in/poster-file/NTQ2OTg3MTU1)
- A follow-up uses a Vision Transformer for preliminary MISO spatial rainfall pattern forecasts — [EGU26-1176](https://meetingorganizer.copernicus.org/EGU26/EGU26-1176.html)

**Extremes, downscaling and gridded prediction**
- Event-level probabilistic extreme-rain prediction over India ("physics-gated latent dynamics") uses ERA5 predictors and IMD gridded rain targets — [arXiv 2602.03166](https://arxiv.org/html/2602.03166) (not fetched; no numbers)
- High-resolution gridded monsoon prediction framed as computer vision, with ERA5 predictors and IMD rain targets — [arXiv 2601.02445](https://arxiv.org/pdf/2601.02445) (not fetched)
- PrecipCorrDiff uses diffusion to downscale IMD rain from 0.25 to 0.0625 degrees. The authors note that standard super-resolution under-represents extremes — [arXiv 2610.19994](https://arxiv.org/html/2610.19994v1)
- IMDAA (12 km) with RF/SVM for extreme-precipitation detection over the Himalaya and North India — [Springer 2026](https://link.springer.com/article/10.1007/s41748-026-01117-3)
- Hourly analysis: IMDAA and other reanalyses miss rare extremes above 10-30 mm/h — [arXiv 2603.24374](https://arxiv.org/pdf/2603.24374)
- An Ashoka University paper compares DL trained on IMD history with HRES. Per the search snippet, the DL forecasts underestimate rainfall — [Ashoka PDF](https://www.ashoka.edu.in/wp-content/uploads/2024/11/Weather_Forecasting-3.pdf). The PDF text could not be extracted, so details are unverified.
- Older baselines (pre-2022, for context only): DL downscaling of ISM rain — [arXiv 2011.11313](https://arxiv.org/pdf/2011.11313); DL ISM rain forecasting — [arXiv 2107.04270](https://arxiv.org/pdf/2107.04270)

**Bay of Bengal / North Indian Ocean cyclones**
- A CNN 24 h track forecast for the Bay of Bengal has a mean track error of **223.69 km** — [JKKNIU PDF](https://jkkniu.edu.bd/uploads/publication/2026-06-10-838d5c_204492301.pdf)
- An LSTM plus ConvLSTM two-branch model (BoB TCs 2013-2023) has a 6 h track error of 169.58 km and an intensity error of 4.68 kt, reportedly falling to 31.01 km with an expanded dataset — [EGU26-19628](https://meetingorganizer.copernicus.org/EGU26/EGU26-19628.html). These are abstract-level numbers.
- LSTM with Cat-Swarm Optimization for BoB track and intensity (2003-2019) — [IEEE Access 2023](https://doaj.org/article/41887da2726640a4b89e3ca77cceb378)
- "Too Rare to Learn": prescribed cyclone tracks *degrade* a Bay of Bengal ocean emulator, because cyclones are too rare in training data — [arXiv 2609.04635](https://arxiv.org/pdf/2609.04635) (title-level only)
- MAUSAM reports that global AI models give divergent cyclone tracks over South Asia — [arXiv 2509.01879](https://arxiv.org/abs/2509.01879)

**Heatwaves:** I found no India heatwave ML forecasting study with numbers in this session.

### Inferences (applicability to our system)
- The suppressed variance and under-predicted extremes in GraphCast and the MAUSAM set are generic to deterministic MSE-trained models. Our heavy-rain CSI and anti-blur scores face the same pressure. These papers support adding spectral or variance-preserving losses (or PARADIS-style advection, which retains spectral amplitude) rather than expecting the architecture alone to fix it.
- The northward-propagation weakness in GraphCast is a reminder that transport and propagation structure are hard for pure ML even at scale. That supports keeping structural advection in our hybrid.
- With about 4,700 training days, cyclones (a handful of BoB events) are "too rare to learn" from data, as 2609.04635 warns. That is a strong argument for a transport operator that generalises (as our hybrid did on Amphan) rather than a pure CNN.
- Onset and MISO work runs at weekly-to-monthly scales and is not directly usable for 10-day daily-mean grading. MISO indices could serve as a slow conditioning input, but that risks leakage or complexity. Low priority.
- We grade against ERA5, so the 15-45% obs-versus-reanalysis gap does not affect scores. It does mean ERA5 rain over India (which is model-generated) is a soft target, so heavy-rain CSI is judged against ERA5's own extremes.

### Gaps
- IITM, IMD and NCMRWF-authored *ML medium-range* (1-10 day) gridded forecast systems with published skill numbers were not found in this session. Only evaluation and collaboration papers were found (for example IITM-coauthored 2607.11905).
- No India heatwave ML study found.
- No BSISO deep-learning study found specifically. The IIT Delhi MISO work is the closest.
- Full numbers for 2602.03767 (onset benchmark), 2602.03166 (extremes) and 2601.02445 were not retrieved.
- No study evaluated ML models with causal shock or perturbation tests over India.

---

## Q4. Public India datasets for augmenting training (pre-2023 data)

### Takeaway
IMDAA (NCMRWF/UKMO 12 km, hourly, 1979-2020) is free for non-commercial research and covers our domain and period. It is the most natural extra source, for example as a second "truth" for pretraining or for checking robustness. IMD 0.25-degree gridded daily rainfall is the standard observational rain dataset. Mixing either one into ERA5-graded training shifts the target distribution, so use them as auxiliary pretraining or validation, not as replacement targets.

### Cited Findings
- **IMDAA**: 12 km, 1-hourly regional reanalysis over India, 1979-2018, extended to Dec 2020. After 2021, "IMDAA-like" products come from NCUM global analysis. It is free and non-commercial with an acknowledgement requirement, in GRIB2 or NetCDF with area subsetting, via the NCMRWF RDS portal — [NCMRWF RDS](https://rds.ncmrwf.gov.in); [NCMRWF TR](https://www.ncmrwf.gov.in/Reports-eng/TR_IMDAA_like_product.pdf)
- IMDAA misses rare hourly extremes above 10-30 mm/h, like other reanalyses — [arXiv 2603.24374](https://arxiv.org/pdf/2603.24374)
- IMD gridded rain at 0.25 degrees is used as the target in several 2025-2026 India ML studies — [arXiv 2602.03166](https://arxiv.org/html/2602.03166); [arXiv 2610.19994](https://arxiv.org/html/2610.19994v1)
- **[unverified this session]** IMD 0.25-degree daily gridded rainfall (Pai et al. 2014, Mausam 65(1)) covers 1901-present over Indian land only, from about 6,955 gauges, distributed by IMD Pune. It is land-only, so it does not cover the Bay of Bengal or Arabian Sea parts of our grid.

### Inferences
- IMDAA's daily means (t2m, 10 m wind, msl, precipitable water, rain) could be regridded to 1 degree. That adds an independent reanalysis of the same 1979-2020 days, which works as data augmentation or reduces overfitting to ERA5 idiosyncrasies. It does not add new days beyond ERA5. Variable naming and units need checking.
- IMD gridded rain is land-only and differs systematically from ERA5 rain. It is useful for validating heavy-rain patterns, but not as a direct training target if grading uses ERA5.

### Gaps
- Whether IMDAA provides total column water vapour and daily-mean fields directly was not checked.
- The IMD gridded rainfall access URL and licence were not verified this session.
- No public "India ML weather benchmark" (a WeatherBench-style regional benchmark) was found.
