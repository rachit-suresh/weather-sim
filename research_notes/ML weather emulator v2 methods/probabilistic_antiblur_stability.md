# Probabilistic training, anti-blur losses and autoregressive rollout stability for ML weather models (2022-2026), mapped to a 32x36 India daily-mean emulator

Context for the report writer: the target system is a 1-degree, 32x36 regional CNN emulator of 6 daily-mean surface variables. It has about 4,700 training days and runs 10 autoregressive 1-day steps. Inference runs on a 4-core CPU in under 5 s and must be deterministic: a zero perturbation must reproduce the output within 1e-3 std. The current loss is latitude-weighted anomaly MSE plus a spatial-std spread loss plus an FFT amplitude loss, trained with a 1-2-4-8 day unroll curriculum. RMSE skill points are already maxed. The open targets are:
- heavy-rain CSI at 10 mm/day;
- day-8 spatial variance between 40% and 250% of observed;
- shock stability (no change larger than 5 std).

Already rejected: rain quantile mapping, post-hoc variance rescaling, damped-persistence blending, and averaged ensembles.

## Q1. CRPS-based training (AIFS-CRPS, FGN, NeuralGCM-ENS, FourCastNet 3, CRPS-LAM, fair/almost-fair CRPS, noise injection)

### Takeaway
Training a noise-conditioned single-pass network with (almost-)fair CRPS on only 2-4 members per sample, plus multi-step rollout fine-tuning, is now the dominant recipe (ECMWF AIFS-CRPS/AIFS-ENS, Google FGN/WeatherNext 2, NVIDIA FCN3, regional CRPS-LAM). Its individual members keep realistic spectra with no lead-time damping, at a cost of one forward pass per member per step. Pointwise CRPS alone tends to *over*-produce small-scale noise. The fixes are a spectral CRPS term (NeuralGCM, FCN3) or a multi-scale CRPS (ECMWF). A single member is sharp but is a sample, not a mean, so its RMSE is higher than the ensemble mean's. That matters for our RMSE-vs-climatology grading (see Inferences).

### Cited Findings
**AIFS-CRPS (ECMWF, arXiv 2412.15832)**
- Trains with the almost-fair CRPS (afCRPS), a linear combination of CRPS and fair CRPS. It "approximately removes the bias … due to finite ensemble size yet avoids a degeneracy of the fair CRPS". Training uses α = 0.95. — [Lang et al. 2024, AIFS-CRPS](https://arxiv.org/html/2412.15832v1)
- Members per training sample: 4 at O96 (~1 degree) and 2 at N320 (~0.25 degree). Members come from independent model instances started from the same deterministic analysis. — [AIFS-CRPS](https://arxiv.org/html/2412.15832v1)
- Noise injection: per-instance Gaussian noise on the processor grid passes through a 2-layer MLP and a layer norm. It then feeds conditional layer norms that replace the standard layer norms in the transformer processor. — [AIFS-CRPS](https://arxiv.org/html/2412.15832v1)
- Rollout curriculum:
  - Stage 1: rollout 1, about 300k iterations.
  - Stage 2: rollout 2, about 60k iterations.
  - Stage 3: rollout 3 to 12 steps (6 h steps, so up to 72 h), about 45k iterations.
  - Stage 4: fine-tuning on the operational analysis with rollout up to 12.

  — [AIFS-CRPS](https://arxiv.org/html/2412.15832v1)
- Results:
  - 50 members for medium range. CRPS and RMSE are better than IFS ENS by 5-20% for upper-air variables. Surface and tropical results are mixed, and the stratosphere is degraded.
  - 8 members for 46-day subseasonal runs, which the paper calls "stable at longer lead times".

  — [AIFS-CRPS](https://arxiv.org/html/2412.15832v1)
- Sharpness: AIFS-CRPS spectra show no damping of small scales with lead time, while the MSE-trained deterministic AIFS progressively loses high-wavenumber energy. Without reference-field truncation, AIFS-CRPS gains *spurious* small-scale energy with lead time. The paper gives no scalar sharpness metric. — [AIFS-CRPS](https://arxiv.org/html/2412.15832v1)
- Cost: one forward pass per 6 h step per member. One member takes about 1 min (O96) or about 4 min (N320) on an A100 for 15 days. — [AIFS-CRPS](https://arxiv.org/html/2412.15832v1)

**Multi-scale afCRPS (ECMWF, arXiv 2506.10868)**
- Pointwise afCRPS cannot tell spatial scales apart. In a 1-D sine test it was invariant to the predicted wavenumber, and AIFS-CRPS showed "more small-scale variability than ERA5", for example noisy 500 hPa contours. — [Lang, Leutbecher & Maciel 2025](https://arxiv.org/html/2506.10868v1)
- The fix: Gaussian-kernel smoothing (σ = 8 grid spacings) splits each field into a large scale and a residual small scale, and an afCRPS is summed per scale. The result is equal CRPS skill ("on top of each other") and better-constrained small-scale spectra, for "only marginal additional cost". Members are evaluated with 8-member ensembles. — [Lang et al. 2025, multi-scale loss](https://arxiv.org/html/2506.10868v1)

**FGN / WeatherNext 2 (Google DeepMind, arXiv 2506.10772)**
- Trained only on per-location (marginal) CRPS, yet captures joint spatial structure. The authors attribute this to a "low-dimensional, globally shared noise source": a 32-dim z ~ N(0, I) fed through one matmul into all conditional layer norms. — [Alet et al. 2025, FGN](https://arxiv.org/html/2506.10772v1)
- Ensemble design: 4 independently trained models × 14 members = 56 members. Training uses N = 2 samples per CRPS estimate with the fair CRPS. Each model has about 180M parameters. — [FGN](https://arxiv.org/html/2506.10772v1)
- AR fine-tuning: 8k steps at 1 AR, 4k at 2 AR, then 1k each at 3-8 AR. The single-step model beats GenCast on 91% of CRPS targets (mean gain 2.7%). With AR training it beats GenCast on 99-99.8% (mean 4%, max 11.5%). — [FGN](https://arxiv.org/html/2506.10772v1)
- Final results vs GenCast:
  - CRPS is better on 99.9% of targets, up to 18% (mean 6.5%).
  - Ensemble-mean RMSE is better on 100% of targets, up to 18% (mean 5.8%).
  - Pooled CRPS improves 7.5-8.7%.

  — [FGN](https://arxiv.org/html/2506.10772v1)
- Sharpness: spectra "generally match" ground truth out to 15 days with no blur-type high-frequency loss. Some fields show extra high-frequency artifacts near 100 km. — [FGN](https://arxiv.org/html/2506.10772v1)
- Cost: a 15-day trajectory takes just under 1 min on a TPU v5p, about 8× faster than GenCast. — [FGN](https://arxiv.org/html/2506.10772v1)
- WeatherNext 2 is built on FGN. — [Google WeatherNext docs (via search)](https://developers.google.com/weathernext/guides/osmodel)

**NeuralGCM-ENS (arXiv 2311.07222)**
- Trained on grid-point plus spectral CRPS. It uses exactly 2 samples per observation, arguing that extra compute is better spent on more initial times. Its ensemble is generated from a single unperturbed initial state. — [Kochkov et al., NeuralGCM](https://arxiv.org/pdf/2311.07222)

**FourCastNet 3 (NVIDIA, arXiv 2507.12144)**
- A composite loss of spatial CRPS plus a spectral CRPS on each spectral coefficient. The spectral term is there because pointwise CRPS only sees marginals. Members keep realistic spectra out to 60 days, and generation is single-step, 8-60× faster than diffusion. FCN3 also notes that CRPS-trained models often fail to give members with physically accurate spectra. — [Bonev et al. 2025, FCN3](https://arxiv.org/html/2507.12144v2); [NVIDIA blog](https://developer.nvidia.com/blog/fourcastnet-3-enables-fast-and-accurate-large-ensemble-weather-forecasting-with-scalable-geometric-ml/)

**CRPS-LAM (regional, arXiv 2510.09484)**
- This is the closest analogue to our setup.
- Architecture: a convolutional U-Net on the MEPS limited-area domain at 10 km. A 32-dim latent z goes through one linear layer into conditional normalization.
- Training: fair CRPS, 1-step training, then 2-step AR fine-tuning.
- Evaluation: 25 members to 57 h.

  — [CRPS-LAM, arXiv 2510.09484](https://arxiv.org/html/2510.09484v2)
- Results:
  - Errors comparable to Diffusion-LAM.
  - Spread-skill comparable to the best baseline, while Diffusion-LAM is more underdispersed.
  - Spectra keep more fine scale than Graph-EFM but lose some of the highest frequencies that diffusion keeps.
  - About 0.5 s per member for 57 h on an A100, about 39× faster than diffusion.

  — [CRPS-LAM](https://arxiv.org/html/2510.09484v2)
- Failure modes:
  - Early-training collapse to near-deterministic output.
  - Artifacts in a few members at long leads, which disappeared once AR training was added.
  - No deterministic MSE baseline and no precipitation results were reported.

  — [CRPS-LAM](https://arxiv.org/html/2510.09484v2)

**Swift (arXiv 2509.25631)**
- Fair-CRPS AR fine-tuning with N = 2 members on a 1-step consistency model. Curriculum K = {1,2}, then 3, then {4,8}. — [Stock et al. 2025, Swift](https://arxiv.org/html/2509.25631v1)

### Inferences
- **Determinism is easy.** With noise entering only as a fixed-seed latent z (for example z = 0, or one fixed draw per member), the network is a deterministic function of its input. The zero-perturbation test then passes exactly, as long as the same z is reused across perturbed and unperturbed runs.
- **Locality caveat (important).** AIFS, FGN and CRPS-LAM inject noise through *conditional layer norm*. A layer/instance norm that normalises over spatial dimensions couples every grid point, which could break our locality test the way variance rescaling did. For our CNN, use per-pixel conditioning (FiLM on channels with no spatial statistics, or GroupNorm-free blocks) or add a spatial noise field directly, rather than any normalisation that pools over the domain.
- **RMSE risk for a single member.** A single member of a calibrated ensemble is a sample. At lead times where skill saturates, its expected MSE approaches about 2× climatological variance, versus 1× for climatology. This is the standard double-penalty argument behind [Subich et al.](https://arxiv.org/html/2501.19374v2), stated as an inference here. Day-8 RMSE skill vs climatology could therefore drop if we ship one member. Mitigations:
  - Use a small α-blend (afCRPS α near 1 still rewards matching the mean).
  - Weight the CRPS loss lightly on top of the existing MSE.
  - Choose z = 0 if the model is trained so that z = 0 is close to the median. This needs an explicit loss term because it is not guaranteed by CRPS training (unverified).
- **Precipitation.** A CRPS-trained member should produce realistic rain intensities, which likely helps 10 mm/day CSI without quantile mapping. This is inferred: no source found reports regional single-member CSI vs MSE.
- **Cost.** At 32x36 with a small CNN, one forward pass per step is milliseconds on CPU. One member × 10 steps is far below 5 s. Training with 2-4 members per sample multiplies training cost by 2-4×.

### Gaps
- Exact afCRPS α used in AIFS-ENS operational versions (2025-2026); only α = 0.95 from the 2024 paper was verified.
- Whether any paper reports RMSE of a *single* CRPS member vs an MSE-trained deterministic model. None found; all papers report ensemble-mean RMSE.
- Whether a "z = 0" or median member can be made to behave like a deterministic forecast. No primary source found.
- NeuralGCM-ENS skill and spectral numbers were not extracted (paper not fetched in full).

## Q2. Diffusion and flow-matching forecasters (GenCast, ArchesWeatherGen, CorrDiff, DYffusion, consistency and one-step models)

### Takeaway
Diffusion and flow-matching models give the sharpest members and strong CRPS. They typically need 20-40 network evaluations (NFE) per step: ArchesWeatherGen uses 25 Euler steps, and the Swift diffusion baseline uses 39 NFE. The 2025-2026 trend is to collapse this to 1 NFE through consistency or one-step flow models (Swift, Tyche) that are then CRPS fine-tuned on rollouts. At that point they are functionally similar to CRPS-trained noise-conditioned networks. With a fixed seed, all of them are deterministic. At 32x36 on CPU, 25 NFE × 10 steps is probably feasible within 5 s for a small CNN, but it is a poor trade compared with a 1-NFE CRPS model.

### Cited Findings
**ArchesWeatherGen (arXiv 2412.12971)**
- Design: flow matching on the *residual* of a deterministic model, r = (x_{t+1} − f_θ(x_t))/σ, conditioned on ArchesWeather-Mx4 at 24 h steps.
- Sampler: 25 Euler steps with a constant step size ("set to 25 based on initial experiments") and an initial-noise scale of 1.05.

  — [Couairon et al. 2024, ArchesWeather/ArchesWeatherGen](https://arxiv.org/html/2412.12971v1)
- Results:
  - CRPS skill is better than IFS ENS on all headline variables and leads, by 5.3% on average.
  - About 4% better CRPS than its own DDPM baseline.
  - Spread-skill about 0.96 at 24 h and about 0.98 later, versus 0.68-0.70 for the DDPM baseline at 24 h.

  — [ArchesWeatherGen](https://arxiv.org/html/2412.12971v1)
- Sharpness: deterministic models "are blurrier than the ground truth" and lose energy with lead time. ArchesWeatherGen activity is "very close to the ERA5 reference", and its ensemble mean stays close to the deterministic prediction at 1 day. — [ArchesWeatherGen](https://arxiv.org/html/2412.12971v1)
- Cost: about 3.5 s per 24 h step on an A100, and about 1 min per 15-day member. Training is about 45 V100-days for the generative part plus about 9 for the deterministic part. — [ArchesWeatherGen](https://arxiv.org/html/2412.12971v1)

**GenCast (Nature 2024, arXiv 2312.15796)**
- A 15-day, 0.25-degree, 12 h-step ensemble of 50+ members in about 8 min on a Cloud TPU v5. The arXiv v1 was at 1 degree, about 1 min per member on a TPU v4. — [GenCast arXiv](https://arxiv.org/html/2312.15796v2); [DeepMind blog](https://deepmind.google/en/blog/gencast-predicts-weather-and-the-risks-of-extreme-conditions-with-sota-accuracy/)
- FGN, a CRPS-trained model, later beats GenCast on 99.9% of CRPS targets while being about 8× faster. — [FGN](https://arxiv.org/html/2506.10772v1)
- In year-long rollouts, GenCast blew up after 71-91 days for most variables and lost seasonality at 79-105 days. — [Lehmann et al. 2026](https://arxiv.org/html/2605.30184)

**Swift, a consistency model (arXiv 2509.25631, NeurIPS 2025 CCAI workshop)**
- Method: a 1 NFE per step consistency model with fair-CRPS AR fine-tuning. The diffusion baseline uses 39 NFE per step (DPM-Solver++ 2S).
- Speed: reported as about 30× faster wall-clock in the text and 39× in the abstract. The 39× matches the NFE ratio.

  — [Swift](https://arxiv.org/html/2509.25631v1)
- Results:
  - Competitive with IFS ENS on most variables and with its own diffusion baseline. Marginally below GenCast.
  - Underdispersive (spread-skill < 1).
  - Z and MSLP spectra drift at high wavenumbers with lead time.
  - Stable out to 75 days at 24 h steps, assessed only qualitatively.

  — [Swift](https://arxiv.org/html/2509.25631v1)

**Tyche (arXiv 2605.06916, 2026)**
- A one-step (1 NFE) conditional flow model with curriculum CRPS rollout fine-tuning, on ERA5 at 1.5 degree and 6 h. It is claimed to match or exceed multi-step generative baselines and IFS ENS, 12-45× faster. Only the abstract was read, so the numbers are not verified. — [Tyche abstract](https://arxiv.org/abs/2605.06916)

**Regional diffusion (Diffusion-LAM)**
- Captures the highest frequencies, but is more underdispersed than the CRPS model and about 39× slower. — [CRPS-LAM paper](https://arxiv.org/html/2510.09484v2)

### Inferences
- **CPU feasibility at 32x36.** This is an engineering estimate, not from sources. If one forward pass of a small CNN takes about 5-20 ms on 4 cores, then 25 NFE × 10 steps = 250 passes takes about 1.3-5 s. That is borderline against the 5 s budget, with no headroom for ensembles. A 1-NFE consistency, one-step flow or CRPS model gives 10 passes.
- **Determinism.** Fixed seeds plus deterministic ODE samplers (Euler or DPM without churn) are reproducible on the same hardware and thread count. But a zero perturbation must give the same noise draw, so the seed must not depend on the input. For SDE or churn samplers, every random call must be seeded.
- **Shock stability.** In the ArchesWeatherGen pattern, the residual sampler sits on top of a deterministic mean model. A shock then propagates mainly through the deterministic backbone, which we already control. The generative part adds sample variance, not a larger shock response. This is an inference.
- **Overall fit.** The ArchesWeatherGen "deterministic + generative residual" pattern maps cleanly onto our system: keep the current CNN, and add a small 1-NFE residual generator trained with CRPS, or a consistency-distilled flow model. That said, a 1-NFE CRPS residual head is the cheaper version of the same idea.

### Gaps
- GenCast sampler details (number of solver steps and NFE per step) were not verified this session. My recollection is a second-order DPM-Solver++ 2S with about 20 steps, consistent with the 39-NFE diffusion baseline in Swift, but this is unverified.
- No source was found for CorrDiff, DYffusion or Stormer-diffusion numbers this session; those were not fetched. Recalled but unverified: CorrDiff (NVIDIA, Mardani et al.) is a regression U-Net mean plus a diffusion residual for km-scale downscaling over Taiwan, the same mean-plus-residual pattern as ArchesWeatherGen. DYffusion (Rühling Cachay et al., NeurIPS 2023) ties diffusion steps to physical time steps to cut sampling cost.
- No paper measured CPU inference at tiny grids.
- Swift and Tyche numeric CRPS values appear only in figures or the full text.

## Q3. Losses against blur (AMSE, spectral and multi-scale losses, adversarial and perceptual losses) and the RMSE-vs-sharpness trade-off

### Takeaway
The strongest deterministic anti-blur evidence is AMSE (Subich et al. 2025). It splits MSE into a spectral amplitude term and a coherence term in which the larger of the two spectra multiplies (1 − coherence). This removes the incentive to damp unpredictable scales. In GraphCast fine-tuning it raised effective resolution from about 1,250 km to about 160 km at 5 days, with CRPS improving and ensemble RMSE roughly unchanged. Our current "MSE + FFT amplitude" loss is a cruder version of this: it penalises amplitude but still lets MSE reward damping, so the two terms fight. Adversarial losses (Kunyu) are claimed to keep RMSE competitive while sharpening, but the evidence is thin and GANs are fragile.

### Cited Findings
**AMSE (Subich et al., arXiv 2501.19374)**
- Definition: per total wavenumber k, AMSE = (√PSD_f − √PSD_a)² + 2·max(PSD_f, PSD_a)·(1 − coherence_k). It is parameter-free and a "drop-in replacement" for MSE, though not a true metric. — [Subich et al. 2025, AMSE](https://arxiv.org/html/2501.19374v2)
- Setup:
  - GraphCast at 0.25 degree, fine-tuned with AR curriculum 1 → 2 → 4 → 8 → 12 steps of 6 h. Batch counts were 25k, 2.5k, 2.5k, 1.25k and 1.25k.
  - About 26.2 A100-days.

  — [AMSE](https://arxiv.org/html/2501.19374v2)
- Effective resolution (25% energy-loss threshold, 5 days):
  - Control: about 1,250 km.
  - AMSE fine-tuned with 12 AR steps: about 160 km.
  - AMSE with only 1 AR step: about 450 km.

  Multi-step training therefore matters as much as the loss. At long leads AMSE-AR12 slightly *overshoots* small-scale energy (amplitude ratio > 1). — [AMSE](https://arxiv.org/html/2501.19374v2)
- RMSE and CRPS (lagged ensemble):
  - CRPS improves consistently: z500 at 5 days goes 84.3 (control) → 82.1 (MSE AR12) → 80.5 (AMSE AR12).
  - Ensemble RMSE "changes little" and eventually becomes better at long leads.
  - No deterministic RMSE delta is reported.

  — [AMSE](https://arxiv.org/html/2501.19374v2)
- Extremes: removes the weak bias in tropical-cyclone intensity, and surface-wind quantiles better match HRES. **Precipitation shows small CRPS gains at long leads and slight degradation at short leads**, which the authors attribute to rain's non-Gaussian distribution. — [AMSE](https://arxiv.org/html/2501.19374v2)
- On limited-area grids, the spherical-harmonic transform is hard. The authors suggest other multiscale decompositions, such as wavelets. They warn that applying the adjustment without a scale decomposition "could let the model hide a lack of fine-scale variability in coarser scales". — [AMSE](https://arxiv.org/html/2501.19374v2)

**FastNet (Met Office/Turing, arXiv 2509.17601)**
- A modified spherical-harmonic loss combined with MSE gives "very similar MSE performance to an MSE-trained model" while significantly improving spectral fidelity. The MSE-only model's high-wavenumber blurring worsens with lead time. — [Dunstan et al. 2025, FastNet](https://arxiv.org/html/2509.17601v1)

**Multi-scale CRPS (ECMWF)**
- Gaussian-filter scale separation with equal skill and better small-scale spectra, at marginal cost. See Q1. — [arXiv 2506.10868](https://arxiv.org/html/2506.10868v1)

**Spectral CRPS (NeuralGCM, FourCastNet 3)**
- See Q1. — [NeuralGCM](https://arxiv.org/pdf/2311.07222); [FCN3](https://arxiv.org/html/2507.12144v2)

**Adversarial: Kunyu (arXiv 2312.08264)**
- Regression plus adversarial loss. The authors claim it stays competitive with HRES on RMSE and ACC while producing sharper fields and better anomaly extremes. Only the search snippet was read, so the numbers are not verified. — [Kunyu](https://ar5iv.labs.arxiv.org/html/2312.08264)
- Radar nowcasting (IRENE) needed a small adversarial weight (w_D = 0.04) and a warm-up before the adversarial term was enabled. This is indirect evidence of training fragility. — [IRENE, arXiv 2609.17175](https://arxiv.org/pdf/2609.17175)

**Mechanism of blur and instability**
- MSE-trained neural PDE solvers are accurate only on a narrow high-amplitude frequency band. Neglected low-amplitude high frequencies degrade long rollouts. PDE-Refiner models all frequencies through diffusion-style refinement and gets longer accurate rollouts. — [Lippe et al., PDE-Refiner, NeurIPS 2023](https://arxiv.org/abs/2308.05732)

### Inferences
- **Highest-value, lowest-risk change for us.** Replace "MSE + FFT amplitude" with AMSE computed on a 2-D DCT of the 32x36 field. A DCT avoids the wrap-around artifacts of a periodic FFT on a non-periodic regional box; this is our adaptation, not from the source. Bin by total wavenumber. Keep the latitude weighting by applying √cos(lat) before the transform.
  - AMSE is deterministic, adds no inference cost, and keeps locality at inference, since it is a training loss only.
  - The AR curriculum we already use (1-2-4-8) is exactly where AMSE gave its gains (AR12 versus AR1).
- **Precipitation.** AMSE's slight short-lead degradation for precipitation suggests applying it to precipitation on a transformed variable (log1p or sqrt), or with a smaller weight. This is an inference.
- **Variance window.** A day-8 variance window of 40-250% is wide. AMSE-AR12's small overshoot (ratio > 1) would sit comfortably inside it.

### Gaps
- No source gives the deterministic RMSE change from AMSE fine-tuning at day 1, 4 or 8. AMSE reports only CRPS and ensemble RMSE.
- No 2022-2026 source found that quantifies perceptual (feature-space) losses for weather.
- No regional or limited-area implementation of AMSE found.
- Adversarial-loss numbers for medium-range weather were not verified.

## Q4. Rollout stability (multi-step fine-tuning, pushforward, noise injection, spectral normalisation, residual/anomaly prediction, Lipschitz constraints)

### Takeaway
Across all major 2023-2026 systems, multi-step (AR) fine-tuning is the universal ingredient: AIFS to 12 steps, FGN to 8, AMSE to 12, Swift to 8, and CRPS-LAM to 2. In CRPS and AMSE models it both stabilises rollouts and is *required* for sharpness gains. The 2026 long-rollout benchmark finds that unstable models amplify high-frequency energy, while stable models act as denoisers. Blurrier models (Aurora, SFNO, Pangu) were the most stable; sharper ones (GraphCast, AIFS, GenCast) blew up within days to months. There is a real tension between sharpness and stability, but over 10 steps the evidence says AR fine-tuning is sufficient.

### Cited Findings
**Long-rollout benchmark (Lehmann et al., arXiv 2605.30184)**
- Setup: 9 models in rollouts of up to 2+ years.

  — [Lehmann et al. 2026](https://arxiv.org/html/2605.30184)
- Blow-up times:

  | Model | Blow-up |
  |---|---|
  | Aurora, SFNO, Pangu | none within 730 days |
  | FourCastNet (v1) | 8-9 days |
  | AIFS | 16 days (MSLP), 41 days (T2m) |
  | GenCast | 71-91 days |
  | GraphCast | 274-470 days |

  — [Lehmann et al. 2026](https://arxiv.org/html/2605.30184)
- Sharpness vs stability: Aurora, SFNO, Pangu and DLESyM are blurrier than ERA5 (small-scale energy ratio < 1). GraphCast and AIFS amplify high-frequency content. — [Lehmann et al. 2026](https://arxiv.org/html/2605.30184)
- Noise probe: with 1σ input noise, Aurora, Pangu and SFNO settle to a constant error, while GraphCast's error grows exponentially. Stable models denoise even random-field initial states within about 50 steps. — [Lehmann et al. 2026](https://arxiv.org/html/2605.30184)
- Time step: 24 h step models kept seasonality through 4-year runs, while 6 h step versions lost it after 244-381 days. Coarser and longer steps favour stability. — [Lehmann et al. 2026](https://arxiv.org/html/2605.30184)
- Not tested: multi-step fine-tuning, residual prediction and spectral filtering. — [Lehmann et al. 2026](https://arxiv.org/html/2605.30184)
- Instability is attributed to spectral bias and the resulting error propagation in autoregressive prediction. — [Chattopadhyay & Hassanzadeh, arXiv 2304.07029](https://arxiv.org/pdf/2304.07029)
- Training noise results are mixed: noise or variational training "can improve" stability, but different levels of white noise added to *labels* did not. — [summarised in Lehmann et al. 2026 search snippet / Chattopadhyay et al.](https://arxiv.org/pdf/2304.07029)
- Spectral normalisation and related architectural changes in Fourier neural operators improved stability up to 400 rollout steps. — [McCabe et al. 2023, Towards Stability of Autoregressive Neural Operators](https://arxiv.org/pdf/2306.10619)
- CRPS-LAM artifacts at long leads in a few members disappeared once 2-step AR training was added. — [CRPS-LAM](https://arxiv.org/html/2510.09484v2)
- FGN's AR fine-tuning raised the share of CRPS targets that beat GenCast from 91% to 99%+. — [FGN](https://arxiv.org/html/2506.10772v1)
- Without reference-field truncation, AIFS-CRPS accumulated spurious small-scale energy with lead time. — [AIFS-CRPS](https://arxiv.org/html/2412.15832v1)
- Swift was stable to 75 days at 24 h steps after CRPS AR fine-tuning, assessed qualitatively. — [Swift](https://arxiv.org/html/2509.25631v1)
- PDE-Refiner gives longer stable rollouts by modelling all frequencies. — [PDE-Refiner](https://arxiv.org/abs/2308.05732)

### Inferences
- Our horizon is 10 steps of 1 day. The benchmarks show blow-up beyond 8 days only in old FourCastNet v1, and our 1-2-4-8 curriculum mirrors what every stable system uses. Stability risk mainly appears when we *add sharpness*:
  - CRPS noise, AMSE overshoot, or diffusion members could inject small-scale energy that grows over 10 steps.
  - The guardrails are to extend the curriculum to 10 steps (full rollout, as AIFS uses up to its full window) and to monitor the day-10 spectrum ratio.
  - A multi-scale or low-pass constraint on the noise path, analogous to AIFS's truncation and the ECMWF multi-scale CRPS, is a further guardrail.
- **Shock test (< 5 std).** A "stable models are denoisers" probe like Lehmann et al.'s is a cheap diagnostic for us: inject 1σ noise and check that the error saturates. Lipschitz control through spectral normalisation of the conv layers directly bounds the per-step amplification of a shock. Its cost to RMSE in weather models is undocumented (gap).
- **Residual/anomaly prediction.** We already predict residuals in anomaly space. ArchesWeatherGen also predicts normalised residuals. No source isolates its stability effect.

### Gaps
- The "pushforward trick" (Brandstetter et al., Message Passing Neural PDE Solvers, ICLR 2022) was not fetched. Recalled, unverified: it unrolls the model without gradients through early steps so that training inputs come from the model's own distribution, a cheaper alternative to full backprop-through-time unrolling.
- No source found quantifying spectral normalisation or Lipschitz constraints on a *weather* model's RMSE.
- "Drift" was not formally defined in the 2026 benchmark.

## Q5. Deterministic post-processing that keeps sharpness without breaking locality

### Takeaway
I found no 2022-2026 primary source that does local variance correction or stochastic-to-deterministic distillation for ML weather forecasts in a way that preserves locality. The literature instead moves the anti-blur fix into *training*: AMSE, spectral or multi-scale CRPS, and CRPS AR fine-tuning. Among published patterns, "deterministic mean + learned generative residual with a fixed seed" (ArchesWeatherGen, CorrDiff-style) is the closest to deterministic post-processing that keeps sharpness.

### Cited Findings
- ArchesWeatherGen adds generated residuals onto a deterministic forecast. Members are sharp and the 1-day ensemble mean stays close to the deterministic prediction. — [ArchesWeatherGen](https://arxiv.org/html/2412.12971v1)
- Generative latent-diffusion post-processing of ensembles was "superior for heavy-to-extreme events" but suboptimal at lower thresholds. — [arXiv 2407.04882, via search snippet](https://arxiv.org/html/2407.04882v1)
- The AMSE authors argue that ensemble generation and sharpness/effective resolution are *distinct* problems, so sharp deterministic forecasts can be had through the loss. — [AMSE](https://arxiv.org/html/2501.19374v2)

### Inferences
- **Fixed-z single-pass stochastic model.** For our constraints (locality, determinism, no rescaling), the "post-processing" option most consistent with the literature is a fixed-z single-pass residual generator. It is a small CNN with *local* conditioning, trained with CRPS on the residual of the current model, evaluated with one fixed latent (or z = 0). It is deterministic, local if the conv receptive field is local, and needs one extra forward pass per step.
- **Distillation.** A deterministic student could be trained to match one fixed-seed member (sample distillation). Because that is a sample, it should inherit sharpness, but it also inherits higher single-member RMSE. No sources were found.
- **Rejected methods.** Both rejected methods (averaging ensembles, variance rescaling) are consistent with the literature diagnosis: ensemble means are blurry by construction (ArchesWeatherGen, AIFS), and global rescaling is not scale-aware (multi-scale CRPS paper).

### Gaps
- No primary source on local (windowed) variance correction for ML weather models.
- No primary source on stochastic-to-deterministic distillation for weather.
- No evidence on whether a fixed-seed member preserves heavy-rain CSI better than an MSE model.

## Q6. Method-by-method fit to our constraints (numbers, cost, CPU, 5 s, determinism, locality)

### Takeaway
Ranked by expected value per unit of risk for our system:
1. AMSE-style loss (2-D DCT) under the existing curriculum, extended to 10 steps.
2. CRPS fine-tuning of a residual head with local fixed-seed noise, possibly with a multi-scale CRPS.
3. A 1-NFE consistency or one-step flow residual model.
4. Multi-step diffusion or flow (25-40 NFE), which is feasible but tight on CPU and offers no clear advantage.
5. Adversarial loss, which has weak evidence and fragile training.

### Cited Findings
| Method | Measured effect (source) | Train cost | Inference NFE per step | Determinism with a fixed seed | Locality risk |
|---|---|---|---|---|---|
| AMSE / spectral-coherence loss | Effective resolution 1,250 → 160 km; CRPS better; ensemble RMSE about unchanged; precipitation slightly worse at short lead ([AMSE](https://arxiv.org/html/2501.19374v2)) | About 1× (FFT per batch) | 1 | Exact (no noise) | None (training-only) |
| MSE + spherical-harmonic loss (FastNet) | MSE about equal; better spectra ([FastNet](https://arxiv.org/html/2509.17601v1)) | About 1× | 1 | Exact | None |
| afCRPS, noise-conditioned (AIFS-CRPS) | 5-20% better CRPS/RMSE than IFS ENS upper-air; no spectral damping ([AIFS-CRPS](https://arxiv.org/html/2412.15832v1)) | 2-4 members per sample (2-4×) | 1 per member | Yes | Conditional layer norm pools over space (risk) |
| Fair CRPS, 32-dim latent (FGN, CRPS-LAM) | CRPS +6.5% over GenCast; regional CRPS about equal to diffusion at 39× speed ([FGN](https://arxiv.org/html/2506.10772v1), [CRPS-LAM](https://arxiv.org/html/2510.09484v2)) | 2 members per sample | 1 | Yes | A global z is fine; normalisation choice matters |
| Multi-scale or spectral CRPS | Same skill, fewer spurious small scales ([ECMWF](https://arxiv.org/html/2506.10868v1), [FCN3](https://arxiv.org/html/2507.12144v2)) | Marginal | 1 | Yes | None |
| Flow matching on the residual (ArchesWeatherGen) | CRPS 5.3% better than IFS ENS; spread-skill about 0.96-0.98 ([ArchesWeatherGen](https://arxiv.org/html/2412.12971v1)) | About 5× the deterministic model | 25 | Yes (Euler ODE) | Low (conv) |
| Consistency / 1-step flow (Swift, Tyche) | About IFS ENS level; 30-39× faster than diffusion; underdispersive ([Swift](https://arxiv.org/html/2509.25631v1)) | Pretraining plus CRPS fine-tuning | 1 | Yes | Low |
| Diffusion (GenCast-style) | Sharp; later beaten by FGN; blew up at 71-91 days in long runs ([Lehmann](https://arxiv.org/html/2605.30184)) | High | About 20-40 (39 in Swift's baseline) | Yes if seeded | Low |
| Adversarial (Kunyu) | Claimed RMSE about equal to HRES, sharper ([Kunyu](https://ar5iv.labs.arxiv.org/html/2312.08264)), unverified | Unstable training | 1 | Exact | None |
| Multi-step AR fine-tuning | Required for AMSE gains (AR1 450 km vs AR12 160 km); removes CRPS-LAM artifacts ([AMSE](https://arxiv.org/html/2501.19374v2), [CRPS-LAM](https://arxiv.org/html/2510.09484v2)) | Grows with unroll length | n/a | Exact | None |

### Inferences
- **CPU budget.** Any 1-NFE option costs about 10 forward passes per 10-day rollout, well inside 5 s. 25-NFE flow costs about 250 passes, probably 1-5 s; this is an estimate, so benchmark before adopting.
- **Determinism checklist for any stochastic option:**
  - Seed derived from a constant, not from input data.
  - Deterministic sampler (no churn).
  - Fixed torch thread count and `torch.use_deterministic_algorithms(True)`.

  Note that the zero-perturbation tolerance (1e-3 std) also absorbs float non-associativity across thread counts.
- **Heavy-rain CSI.** The most plausible lever without quantile mapping is a sharp member from CRPS or a generative residual, trained on transformed precipitation. AMSE alone may not help rain (per Subich et al.). This is inferred, with no direct regional evidence.
- **RMSE protection.** The main risk to already-maxed RMSE points is single-member double-penalty at day 8. Keep MSE as the primary loss and add AMSE or CRPS as weighted auxiliaries, then sweep the weight against day-8 RMSE skill. FastNet's "MSE + spectral" combination kept MSE about equal.

### Gaps
- No measured CPU timings for any of these models at about 1,000-cell grids.
- No study reports heavy-rain CSI for single CRPS or generative members versus MSE models at daily, 1-degree resolution.
- Tyche and Kunyu figures are abstract- or snippet-level only.
