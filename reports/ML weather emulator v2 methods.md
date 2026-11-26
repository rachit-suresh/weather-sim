# Teach the emulator to move weather

> **Data-rule correction (added after review).** The track allows only the provided `india_climate_train.nc`
> (2005-2021, 32 x 36). Every suggestion below that needs outside data is OUT: extra ERA5 years (1979-2004, 2022),
> IMDAA / IMD datasets, pre-training on a larger box, distillation from global models, and a 2022 test year.
> E1 becomes: train the final model on all provided years 2005-2021, unroll to 10 days, up-weight cyclone days,
> and a warm-anchored climatology from a per-cell trend fitted on 2005-2021 only. E6 keeps learned edges and drops
> large-box pre-training (random crops inside the 32 x 36 grid are still allowed). E2-E5, E7, E8 are unaffected.


The 2022-2026 methods most likely to improve v2 are not new backbones. They are three targeted changes. The first is a **structural transport operator inside a single model**: neural semi-Lagrangian advection with operator splitting, as in PARADIS. The second is an **explicit moisture-flux-convergence pathway into the rain head**. The third is a **training-recipe upgrade**: more ERA5 years with a climatology re-centred on the warm end of the record, a soft-CSI term at 10 mm/day, and an AMSE-style spectral loss. v1 already scores 68.0/70 with RMSE points maxed, so the auto-graded headroom is only about 2 points, almost all of it in shocks (13.1/15). The bigger prize is robustness. Heavy-rain CSI sits right on the full-points line, and 2023-24 are warmer than any training year, a shift that makes published ML models behave like a climate 15-20 years older. The cheapest high-value experiments change only data and losses and leave inference untouched. The large architectural bet comes after them: it should make cyclones move without the output blend that cost shock points. Global attention, Fourier/spectral mixing, multi-step diffusion samplers and foundation-model distillation each fail the locality, CPU or data-rule constraints, so drop them. One caveat covers everything here: nearly all the method evidence is global-scale and much of it is 2025-2026 preprints. No paper tests a daily-mean, 1-degree regional emulator or causal shock tests. Every transfer claim below is therefore a hypothesis for the local grader to settle, over several seeds, because the margins are small.

## Two points of headroom, but the hidden warm years put more at risk

v1's score sheet shows where the remaining points are. RMSE skill is capped at 0.12 and already full. The day-8 variance window (40-250%) is wide. Shocks leave 1.9 points on the table. Heavy-rain CSI of 0.10-0.11 earns full points only because the threshold is 0.10: a 10% relative drop on unseen years would cost points. So v2 should be judged on four things: shock points, CSI margin, robustness to the warmer 2023-24 test years, and not regressing anywhere else.

The warm-year risk is well documented. FourCastNet, Pangu and ACE2 all produce cold-biased land temperatures that **resemble climates 15-20 years before the forecast period**, and the bias is worst for the hottest predictions ([Landsberg & Barnes, GRL 2026](https://arxiv.org/html/2509.22359v2)). Over 10-day forecasts, Pangu drifts by about **-0.03 K/day in 2023**. The authors attribute the drift to models relaxing toward their training climate ([Rackow et al. 2024](https://arxiv.org/html/2409.18529)). v1 has this exact mechanism: it predicts anomaly changes relative to a 2005-2017 climatology, so any damping pulls 2023-24 forecasts toward 2011-ish conditions. Two remedies have evidence. ArchesWeather finishes training on recent years to reduce distribution-shift error ([Couairon et al. 2024](https://arxiv.org/html/2405.14527)). Emulators also generalise to a warmer climate when fine-tuned on even small amounts of data near the test distribution ([arXiv 2405.18585](https://arxiv.org/html/2405.18585)).

More data is also the best-evidenced fix for the small-sample problem. MET Norway's Bris pre-trains on **43 years of ERA5 and fine-tunes on only 3.3 years** of regional analyses. It beats the operational MEPS control and ensemble mean on 2 m temperature RMSE ([AIES 2026](https://journals.ametsoc.org/view/journals/aies/5/2/AIES-D-25-0001.1.xml)). Aurora reports about **25% better results from pre-training than from training from scratch** ([Bodnar et al.](https://arxiv.org/html/2405.13063)). The rules allow any public data before 2023, and ERA5 1979-2022 gives about 16,000 daily samples, 3.4 times v1's set.

Naive implementation hides a trap. A 1979-2022 mean climatology is centred near 2000, so it is colder than v1's 2005-2017 baseline. Using it would *worsen* the warm drift. The climatology should instead be a per-cell, per-day-of-year trend fit evaluated at a fixed late date (2022, or the middle of the expected test window), baked into the climatology arrays. This uses no post-2022 data. It also needs no year input, which the model does not have. The tempting alternative of anchoring forecasts to the domain-mean input anomaly should be avoided. A domain mean couples every cell to every shock box, and it would fail the locality tests the same way v1's rejected global variance rescaling did.

## Local CNNs cannot carry a cyclone past their receptive field

Amphan staying at 13.5N is a known failure of convolutional architectures, not a tuning problem. In the PARADIS idealised test, a **3-layer U-Net with about 44,000 parameters could not move a Gaussian tracer beyond its receptive field, while a 20-parameter neural semi-Lagrangian layer moved it accurately** ([Pereira et al. 2026](https://arxiv.org/html/2601.21151v1)). Amphan moved roughly 2-3 cells per daily step. A local CNN trained with MSE on a handful of cyclone days lowers its loss by holding a smeared low in place and letting it decay. That also explains why rain lingers after the storm has gone.

The model built on this operator is credible at our resolution. At 1 degree, PARADIS reported 10-day z500 RMSE of 727 against GraphCast's 729 in its first preprint version, and cyclone position errors over more than 80 storms in 2020 that were lower than or comparable to GraphCast, Pangu and HRES ([arXiv 2601.21151v1](https://arxiv.org/html/2601.21151v1)). The revised version is more modest. It beats GraphCast at 6 h but trails at 240 h (806 vs 732), and the authors say it is "not uniformly RMSE-optimal", though it keeps spectral amplitude better than GraphCast, FuXi and Baguan ([arXiv 2601.21151v2](https://arxiv.org/html/2601.21151v2)). The fair reading is that structural advection buys track skill and sharpness at roughly equal RMSE, which is the trade v1 needs. Two other results point the same way. ECMWF found that rollout training made AIFS tropical-cyclone tracks much better, mainly by shrinking a slow-propagation bias ([Lang et al. 2024](https://arxiv.org/html/2406.01465)). NowcastNet's design (a neural advection "evolution" network plus a learned generator) was motivated by the observation that purely data-driven models fail to obey advective conservation ([Zhang et al., Nature 2023](https://www.nature.com/articles/s41586-023-06184-4.pdf)).

PARADIS's design also explains why v1's physics hybrid lost 2-3 points. The hybrid advects the *raw anomaly* of every variable in one bilinear step capped at 4 cells, then corrects afterward. PARADIS works differently in three ways. It advects a **learned low-dimensional latent subspace**, so the model chooses which features transport well. It blends advection with a **learned per-channel weight**. And it runs a **Lie-Trotter split of advection, diffusion and reaction inside every substep**, using bicubic sampling to limit numerical diffusion ([arXiv 2601.21151v2](https://arxiv.org/html/2601.21151v2)). That lets the model fall back to CNN-like local updates wherever transport is the wrong operator. It is also a single model, which removes the output blend that cost v1 a shock point on test years. NeuralGCM supports the broader pattern: a fixed dynamical core plus learned physics, trained end to end, is competitive with pure ML for 1-10 day forecasts ([Google Research](https://research.google/pubs/neural-general-circulation-models-for-weather-and-climate/)).

The "spreads outward" failures need one more ingredient. A semi-Lagrangian step only *moves* an anomaly. ClimODE models weather with the full continuity equation, which has a transport term and a divergence (compression) term ([Verma et al., ICLR 2024](https://arxiv.org/abs/2404.10024v1)). PARADIS adds an explicit diffusion block ([arXiv 2601.21151v2](https://arxiv.org/html/2601.21151v2)). v1's hybrid has neither, so it displaces a Bay of Bengal anomaly without spreading it. A per-substep Laplacian with a learned, non-negative diffusivity field is the cheapest deterministic, local fix.

## Rain must see moisture convergence, and the loss must see the threshold

v1's heavy-rain weakness is the standard behaviour of deterministic models, not something specific to it. Seven global AI models evaluated over South Asia all **under-predict extreme precipitation** ([MAUSAM, arXiv 2509.01879](https://arxiv.org/abs/2509.01879)). An IITM-coauthored study of GraphCast over the Indian monsoon finds daily correlation with IMERG of 0.89-0.92 but a **spectral variance ratio near 0.63**, with under-represented extremes ([arXiv 2607.11905](https://arxiv.org/abs/2607.11905)). (An earlier version described stronger suppression; the latest version's figures are used here.) Scale and architecture do not remove this, so the fix has to come from the objective and the rain pathway.

The most direct lever is to optimise the graded metric. Ko et al. replaced CSI counts with differentiable approximations and trained on the negative score. They report **up to 43.6% higher heavy-rain CSI** than focal or cross-entropy losses, attributing the gain to easing class imbalance ([arXiv 2202.10555](https://arxiv.org/pdf/2202.10555)). That result comes from hourly radar nowcasting, not daily 1-degree reanalysis, so expect a smaller gain. Even so, a soft-CSI term at 10 mm/day attacks exactly the threshold v1 sits on. Its known side effect is over-forecasting, and a 2026 post-processing study found a CSI term caused underdispersion ([Weather and Forecasting 2026](https://journals.ametsoc.org/view/journals/wefo/41/2/WAF-D-24-0199.1.xml)). Frequency bias must therefore be tracked alongside CSI. Likelihood losses are the second lever. Bernoulli-gamma negative log-likelihood beats MSE across precipitation levels in CNN downscaling ([arXiv 2411.05850](https://arxiv.org/pdf/2411.05850)), and a Tweedie loss raised 99th-percentile recall from about 0.50 to 0.60 ([arXiv 2509.08369](https://arxiv.org/html/2509.08369)). Both fit v1's two-part head without changing inference.

The wind-shock failure is a coupling problem: v1's rain head can simply ignore the wind channels. Evidence that convergence should be explicit is consistent. In MZ-Rain, **removing the moisture-flux-convergence term dropped CSI from 0.324 to 0.296**, and convergence had the strongest link to future rain among budget terms ([arXiv 2609.04864](https://arxiv.org/pdf/2609.04864)). A daily-rain deep-learning study in a monsoonal region found vertically integrated moisture-flux divergence was the single most important predictor ([JMG BMKG](https://jmg.bmkg.go.id/jmg/index.php/jmg/article/download/1194/511)). NeuralGCM only produced realistic precipitation and extremes after precipitation and evaporation became explicit outputs ([Yuval et al.](https://arxiv.org/abs/2412.11973?context=cs)).

For v1's variables, the practical form is an input channel C = -∇·(TCWV·V), computed with a 3x3 stencil, plus the input-day TCWV tendency, both fed to the rain head. V should be a small learned "effective steering wind" rather than raw 10 m wind, because moisture transport follows roughly 850 hPa flow, for which 10 m wind is only a proxy. Weakening the Arabian Sea westerlies then reduces convergence at the Western Ghats and lowers rain *by construction*.

Any budget closure should be a soft penalty, not a hard projection. ACE2's exact conservation corrector eliminated negative precipitation but **raised 10-day precipitation RMSE by 2.46%** in a small pilot ([arXiv 2610.23561](https://arxiv.org/pdf/2610.23561)). In an emulator study, closing the moisture budget by rescaling evaporation produced a **wrong-sign response** ([arXiv 2602.16090](https://arxiv.org/pdf/2602.16090)). A wrong-sign response is exactly what a causal shock test would punish. ERA5 daily means also do not close the budget exactly, so a hard constraint would impose that error on the model.

## Sharpness belongs in the loss; global mixers and samplers fail the constraints

v1 combines MSE with an FFT-amplitude penalty, and the two terms pull against each other: MSE rewards damping unpredictable scales while the amplitude term fights it. AMSE resolves this by splitting the error per wavenumber into an amplitude term and a coherence term. Fine-tuning GraphCast with it raised **effective resolution at day 5 from about 1,250 km to about 160 km**, improved CRPS, and left ensemble RMSE roughly unchanged ([Subich et al. 2025](https://arxiv.org/html/2501.19374v2)). Two details matter for v2. First, the gain depended on multi-step training: AMSE with one rollout step reached only about 450 km. Second, precipitation degraded slightly at short leads, so rain needs a transformed variable or a lower weight ([Subich et al.](https://arxiv.org/html/2501.19374v2)). FastNet independently found that MSE plus a spectral term kept "very similar MSE performance" with much better spectra ([Dunstan et al. 2025](https://arxiv.org/html/2509.17601v1)). On a regional box, a 2-D DCT is the natural transform. The AMSE authors warn that applying the adjustment without a scale decomposition lets a model hide missing fine-scale variance in coarser scales.

CRPS-trained noise-conditioned models (AIFS-CRPS, FGN, regional CRPS-LAM) are the field's main sharpness tool. They fit v2 worse than they first appear. FGN beats GenCast on 99.9% of CRPS targets ([Alet et al. 2025](https://arxiv.org/html/2506.10772v1)), and CRPS-LAM matches diffusion at about 39 times the speed on a regional U-Net ([arXiv 2510.09484](https://arxiv.org/html/2510.09484v2)). The problems for v2 are these:

- **Locality.** These models inject noise through conditional layer norms. Any normalisation that pools over the domain couples every grid point, which is a locality hazard.
- **RMSE.** A single member is a sample, not a mean. At day 8 its expected error approaches twice the climatological variance, which threatens the maxed RMSE points. This is an inference; no paper reports single-member RMSE.
- **Training failure.** CRPS-LAM also saw early collapse to near-deterministic output ([arXiv 2510.09484](https://arxiv.org/html/2510.09484v2)).

Diffusion and flow matching are worse still. ArchesWeatherGen uses **25 Euler steps per forecast step** ([Couairon et al.](https://arxiv.org/html/2412.12971v1)), and Swift's diffusion baseline uses **39 network evaluations** ([Stock et al. 2025](https://arxiv.org/html/2509.25631v1)). By our estimate, 25 evaluations over 10 steps is borderline for 5 s on four cores. And with about 4,700 samples there is little data to learn a distribution.

Rollout stability is not the binding risk over 10 daily steps. In a nine-model long-rollout benchmark, only FourCastNet v1 blew up within 10 days. The sharper models (GraphCast, AIFS, GenCast) were less stable than blurrier ones, and 24 h-step models were the most robust ([Lehmann et al. 2026](https://arxiv.org/html/2605.30184)). Two cheap guardrails follow: extend the curriculum to the full 10 steps, and monitor the day-10 spectrum whenever a change adds sharpness. Every major system ends training with multi-step fine-tuning: AIFS to 12 steps, FGN to 8, AMSE to 12 ([AIFS-CRPS](https://arxiv.org/html/2412.15832v1)).

The locality tests rule out much of the global-model toolbox. GraphCast's multimesh edges span thousands of kilometres in one message-passing step ([Lam et al.](https://arxiv.org/pdf/2212.12794)). SFNO mixes in spherical-harmonic space ([Bonev et al.](https://arxiv.org/abs/2306.03838)). Stormer uses full self-attention over all tokens ([Nguyen et al.](https://arxiv.org/html/2312.03876)). All three put a shock's influence across the 32x36 box on day 1, which is consistent with the global spread and unstable shocks of the FNO in v1's family sweep. Foundation models are no better. Their input contracts need full 3-D atmospheric states, they are 10-10,000 times larger than v1, and the training periods of their weights cannot be checked against the pre-2023 data rule, so even offline distillation carries leakage risk.

## Without boundary inflow, edges must be learned, not forced

Every published ML limited-area model gets lateral information from outside, whether as a forced boundary ring (neural-lam), coarse global conditioning (YingLong, StormCast) or a stretched global grid (Bris). Removing that information **"significantly degrades" skill, and models diverge from truth within hours** at km scale ([arXiv 2504.09340](https://arxiv.org/html/2504.09340v1)). The same paper found **400 km of boundary was as good as 800-1,600 km**, about two steps' worth of feature travel. YingLong found that inserting boundary values abruptly at each step creates discontinuities at the junction ([arXiv 2401.16254](https://arxiv.org/pdf/2401.16254)). No 2022-2026 paper studies a model with *no* boundary input, which is v1's situation, so this part of v2 is empirical.

The geometry makes the problem significant. Daily-mean surface signals travel roughly 4-12 one-degree cells per day (our estimate). By day 3-4 most of the box therefore depends on information that was outside it at the start. Near the edges after day 2, the MSE-optimal forecast is close to climatology plus a damped anomaly. v2 can approach that without external data in three ways:

- **Zero padding in anomaly space.** This means "climatology outside the domain", whereas replicate padding persists edge values and reflect padding invents mirrored inflow.
- **An edge-distance input channel.** The network learns how far to trust its own dynamics near each edge.
- **Pre-training on random 32x36 crops of a much larger ERA5 box.** Window interiors near edges then learn what inflow looks like.

The India regional benchmark offers some reassurance on architecture: a plain UNet stays "close to the top" among 25-35M-parameter models on IMDAA data ([arXiv 2509.00653](https://arxiv.org/html/2509.00653v1)). Note that this benchmark supplied true boundaries. A hard Davies-style relaxation ring toward climatology is cheap but risky here, because the Arabian Sea shock box lies against the western edge and a ring would damp exactly the response the shock test measures. The semi-Lagrangian design handles inflow neatly: departure points that fall outside the domain sample zero anomaly, so climatology flows in automatically and with the right timing.

## Prioritised v2 experiment plan

**Protocol first.** The score margins are small, so a harness change comes before any method.

- **Splits.** Keep v1's 2018-19 validation and 2020-21 test for comparability. Add **2022 as a warm out-of-sample year** while experiments train only on years up to 2017.
- **Seeds and intervals.** Run every candidate with 3 seeds. Report the mean and the worst seed, with day-block bootstrap intervals on CSI.
- **New diagnostics:**
  - Amphan centre latitude on the landfall day (truth 20.5N)
  - "spreads outward" pass count over 20 windows instead of 5
  - the sign and size of Western Ghats rain change under the Arabian Sea -40% wind shock
  - day-8 domain-mean T2m and TCWV bias on 2022
  - RMSE in the outer 4-cell edge band at days 4 and 8
- **Every-candidate checks.** Zero-shock determinism and the 5 s CPU budget.
- **Adoption rule.** Adopt a change only if its mean automatic score on both hold-outs stays within 0.2 of the current best and its target diagnostic improves by more than the seed spread.
- **Final model.** Retrain the frozen recipe once on 1979-2022.

In the cost column, "R" means one v1-sized training run on the current hardware.

| Rank | Experiment | Weakness targeted | Engineering + compute | Expected effect on the local grader |
|---|---|---|---|---|
| E1 | 1979-2022 data, trend-centred climatology, unroll to 10 | Small data, warm test years | 0.5-1 day + ~3.5R per seed | Smaller warm drift; CSI and shock margin from more cyclone and monsoon samples |
| E2 | Soft-CSI loss at 10 mm/day | Heavy rain at threshold | 2-3 h + ~7R | CSI from ~0.10 to 0.12 or more |
| E3 | Moisture-flux-convergence rain pathway | Wind-to-rain coupling | 1 day + ~4R | Correct-sign wind-shock rain response; +0.5-1 shock point |
| E4 | AMSE on a 2-D DCT replacing MSE + FFT amplitude | Blur, shallow cyclones | 0.5 day + ~3R | Day-8 variance nearer 100%, RMSE unchanged |
| E5 | Bernoulli-gamma or soft-gate rain head | Heavy rain | 0.5-1 day + ~4R | Further CSI margin; may substitute for E2 |
| E6 | Edge channel, zero padding, then large-box crop pre-training | No inflow | 0.5 day + 3R; crops +2 days + 5-10R | Lower edge-band RMSE, steadier Arabian Sea shocks |
| E7 | Unified neural semi-Lagrangian operator-split model | Static cyclones, no spreading | 3-5 days + 10-15R | Amphan tracked without score loss; more spread-outward passes |
| E8 | Fixed-latent CRPS residual head (only if E2/E4/E5 fall short) | Blur, CSI | 3 days + 2-4x cost per run | Uncertain; RMSE risk |

### E1: More years, a warm-anchored climatology and a full-length unroll

**Hypothesis.** 3.4 times more data reduces overfitting. A climatology anchored at the warm end of the record removes the pull toward a 2011-like climate that the literature predicts for 2023-24.

**Change.**
1. Download ERA5 daily means for the six variables over the box for 1979-2022.
2. Fit a per-cell, per-day-of-year linear trend (harmonics in day-of-year times a linear term in year) and evaluate it at the end of the training period. Store that as the anomaly baseline. Do this in the offline build script, so no dates or dataset names enter the served package.
3. Train on 1979-2017 for experiments, with a final 2-epoch fine-tune on 2010-2017.
4. Extend the unroll curriculum from 1-2-4-8 to 1-2-4-8-10.
5. Up-weight Bay of Bengal and Arabian Sea cyclone days by 2-3x, as AIFS-TC up-weights rapid-intensification cases ([ECMWF](https://ecmwf.int/en/about/media-centre/aifs-blog/2026/ai-tropical-cyclone-forecasts)).

Pre-1979 ERA5 has fewer observations, so leave it out of the first pass.

**Success.** Mean automatic score at or above v1 on both hold-outs. Day-8 T2m domain-mean bias on 2022 cut by at least half against v1. CSI at least 0.11 in the worst seed.

**Cost.** About a day, plus roughly 10R for 3 seeds.

**Risks.** None to locality, determinism or CPU, because inference is unchanged. The main risk is that a longer record dilutes 2005-2017-specific skill. The late fine-tune is the hedge.

### E2: A soft-CSI term at the graded threshold

**Hypothesis.** v1's 3x heavy-rain weight is a blunt proxy. Optimising a differentiable CSI at 10 mm/day moves the score off the threshold.

**Change.** Add λ·(1 − TP/(TP+FP+FN)) to the loss, where hit probability = sigmoid((rain_mm − 10)/τ), τ = 1.5 mm/day, and counts are summed over cells and unrolled days. Compute it on the soft-gated amount (p·amount) during training, because the hard straight-through gate gives poor gradients near the threshold. Sweep λ ∈ {0.05, 0.1, 0.2} with 2 seeds each, then confirm the winner with a third seed.

**Success.**
- CSI at least 0.12 on both hold-outs in every seed.
- Frequency bias between 0.8 and 1.5.
- RMSE points unchanged.
- Day-8 rain variance inside the window.

**Cost.** A few hours plus about 7R.

**Risks.** Over-forecasting and a small rain-RMSE cost; neither affects locality, determinism or CPU.

### E3: Put moisture-flux convergence on the rain head's critical path

**Hypothesis.** Rain ignores wind because nothing structurally links them. An explicit convergence input makes rain respond to wind shocks with the right sign, and lets the rain response propagate along the flow.

**Change.**
1. Compute V_eff with a learned 1x1 map from (u10, v10, ∂msl/∂x, ∂msl/∂y).
2. Compute C = −∇·(TCWV·V_eff) with centred differences, plus ΔTCWV across the input days. Concatenate both into the rain head's input. In the first variant, concatenate them into the trunk as well.
3. Add a soft penalty, λ_b·|P − softplus(a·C − b·ΔTCWV + c)|² on wet cells, with per-cell learned a, b, c to absorb evaporation and the 10 m vs 850 hPa mismatch. Use λ_b = 0.05 and never hard-project.
4. Add a regression test: a finite-difference shock on V_eff must produce ∂rain/∂C > 0 over the Western Ghats.

**Success.**
- Arabian Sea −40% wind shock lowers Western Ghats rain in all 5 standard windows.
- "Spreads outward" passes in at least 4 of 5 windows (and at least 80% of the 20-window set).
- Shock score at least 13.8/15.
- CSI not lower.

**Cost.** About 1 day plus 4R.

**Risks.**
- The stencil adds one cell to the per-step receptive radius (7 → 8). Re-check the day-1 three-cell test.
- Divergence at the western edge uses zero-padded anomalies.
- The convergence proxy is biased, which is why the penalty stays soft.

### E4: Replace the amplitude penalty with AMSE

**Hypothesis.** Removing the MSE-versus-amplitude tug-of-war gives sharper day-8 fields and deeper lows at equal RMSE.

**Change.** Compute a 2-D DCT-II (two small matrix multiplies) of each latitude-weighted anomaly field. Bin by total wavenumber and apply the AMSE amplitude-plus-coherence formula per bin, replacing MSE + FFT amplitude for t2m, u, v, msl and tcwv. For rain, apply AMSE in sqrt space at 0.3 weight, alongside the existing loss.

**Success.**
- Day-8 variance ratio between 70% and 130% for every variable.
- RMSE points unchanged.
- Amphan day-3 central pressure deeper than v1.
- No day-10 spectral overshoot above 1.5.

**Cost.** Half a day plus 3R.

**Risks.** Training-only, so no locality, determinism or CPU impact. Possible short-lead rain degradation, as Subich et al. saw.

### E5: A likelihood-trained rain amount head

**Hypothesis.** Weighted MSE in sqrt space still regresses intensities toward the mean. A Bernoulli-gamma (or Tweedie) likelihood keeps the tail.

**Change.**
1. Train occurrence with BCE and amount with gamma negative log-likelihood (or a single Tweedie loss with power 1.5).
2. Use a soft gate in training and the hard gate at inference.
3. Output the gamma mean, or a fixed 0.6-0.7 quantile only where p > 0.5, and compare both against the grader.
4. Run it alone and stacked on E2.

**Success.** CSI gain over the best of E1/E2, with rain RMSE within one seed spread.

**Cost.** Half to one day plus 4R.

**Risks.** Gamma NLL can be unstable at tiny amounts, so floor the amount at 0.1 mm. No constraint risk.

### E6: Learned edges, then crop pre-training on a large box

**Hypothesis.** Edge-band errors dominate day 4-10 RMSE and destabilise edge-adjacent shocks. Teaching the model its distance to the edge, and showing it inflow during pre-training, helps both.

**Change.**
- **Stage A.** Switch all convolutions to zero padding in anomaly space (if not already). Add a static edge-distance channel (cells to the nearest edge, capped at 8) and a lead-day channel.
- **Stage B.** Download the same variables for 20S-50N, 30-130E and pre-train on random 32x36 crops with latitude, orography and land-sea-mask channels (no lat/lon index channels). Then fine-tune on the target box.
- **Avoid a hard relaxation ring** unless Stage A fails, and if one is used, exclude the Arabian Sea edge.

**Success.**
- Outer-band RMSE at day 8 down at least 5% with interior RMSE unchanged.
- Arabian Sea shock windows no worse.

**Cost.** Stage A takes half a day plus 3R. Stage B takes about 2 days plus 5-10R.

**Risks.** Crop pre-training conflicts with absolute-position channels if v1 relies on them. Inference is unchanged.

### E7: One model that both advects and corrects

**Hypothesis.** A PARADIS-style operator split keeps v1's CNN accuracy and adds structural transport. That fixes Amphan without a blend, and adds the diffusion that "spreads outward" needs.

**Change.**
1. Encode the state plus statics with a 1x1 conv to 24 channels.
2. Run K = 3 substeps per day. Each substep has four parts:
   - a velocity net (depthwise-separable 3x3) predicting displacement for an 8-channel projected subspace, initialised from smoothed 10 m wind and clipped to 1 cell per substep;
   - bicubic `grid_sample` with zero padding;
   - a learned per-channel blend weight α, initialised near 0 so the model starts as the CNN;
   - a Laplacian diffusion step with softplus diffusivity, then a pointwise reaction MLP.
3. Decode with v1's two-part rain head.
4. Warm-start the reaction and decoder layers from v1's CNN and train with E1's curriculum.
5. Clamp rain at zero after bicubic sampling.

**Success.**
- Automatic score within 0.3 of the best CNN on both hold-outs.
- Amphan landfall-day centre within 2 cells of 20.5N.
- "Spreads outward" passes in at least 4 of 5 windows.
- Shock score at least v1's.

**Cost.** 3-5 days of engineering plus 10-15R, including a K ∈ {2, 3, 4} sweep.

**Risks.**
- **Locality.** The day-1 three-cell rule bounds total displacement, so cap it at 3 cells/day; Amphan needed 2-3.
- **Determinism.** Fix the thread count and keep `grid_sample` in CPU float32.
- **Runtime.** About 30 sampling calls on a 32x36 grid is far below 5 s.
- **Training.** Learned velocity fields can destabilise early training. The α ≈ 0 initialisation is the hedge.

### E8: Fixed-latent CRPS residual head, only if needed

**Hypothesis.** If E2, E4 and E5 leave CSI or variance marginal, a noise-conditioned residual trained with almost-fair CRPS supplies sharp, realistic intensities.

**Change.**
1. Add a small local CNN that predicts a residual from (state, z).
2. Feed z as a spatial noise field or per-pixel FiLM. Use no conditional layer norm and no domain-pooled statistics.
3. Train with afCRPS (α = 0.95) on 2 members, with MSE kept as the primary loss.
4. Ship one fixed z (constant seed, independent of the input) so that zero shock equals no shock.

**Success.** CSI or variance gain with day-8 RMSE skill unchanged.

**Cost.** About 3 days, with each training run 2-4 times as expensive.

**Risks.**
- The single-member double penalty on day-8 RMSE.
- Collapse to near-deterministic output.
- Locality breakage if any normalisation pools over space.

**Deliberately excluded:**
- Multi-step diffusion or flow samplers: CPU-borderline and data-hungry.
- FNO, ViT, multimesh or Swin backbones: they fail the locality tests.
- Seed-averaged ensembles: already rejected, and blurry by construction.
- Distillation from AIFS, GraphCast or Aurora: data-rule exposure and input-contract mismatch.
- IMDAA or IMD gridded rain as targets: they shift the target away from the ERA5 the grader uses. IMDAA could serve later as auxiliary pre-training.

## Conclusion

The evidence changes what v2 should aim at. v1's remaining failures are not about capacity. They are missing *operators*: a transport operator for cyclones, a divergence and diffusion operator for spreading, and a convergence operator linking wind to rain. Literature at every scale shows that local convolutions and MSE cannot invent these from about 4,700 samples. v1's physics hybrid lost points not because physics hurt, but because it advected the wrong quantity, corrected too late, and was integrated by blending outputs. PARADIS shows that fixing those three things inside one model is feasible.

The second shift is about risk rather than skill. With only two automatic points left, the largest expected change in v2's real score comes from the hidden warm years and from CSI sitting on its threshold. The best defence costs almost nothing: more years, a climatology anchored at the warm end of the record rather than a naive long-term mean, and a loss that sees the 10 mm/day line. The research also leaves a gap v2 can fill. No published regional emulator has been evaluated with causal shock tests or without boundary inflow, so the expanded local harness is a prerequisite for every experiment above, not optional polish.
