# Model families

Our data is one weather "frame" per day: a grid of 32 x 36 cells with 6 numbers per cell
(`t2m, tp, u10, v10, msl, tcwv`) plus 2 fixed maps (land/sea, height). The model sees 3 frames and
must produce the next 10.

## Are these "image models"?

Partly. A regular lat/lon grid with channels is shaped exactly like an image (height x width x
channels), so image architectures (CNN, U-Net, ViT) run on it unchanged. But it is **not** a
photo:

| Photo | Our weather grid |
|---|---|
| 3 channels (RGB), all the same kind | 6 channels, each a different physical quantity with its own units and range |
| Objects can be anywhere, scale varies | Physics is local: air at one cell only affects nearby cells in one day |
| Edges of the photo do not matter | Edges matter: weather flows in from outside the box and we never see it |
| Output is a label or a picture | Output is fed back in as input 10 times, so small errors grow |
| Millions of training images | About 6,200 days |

So when this doc says "image model" it means "a network that treats the grid like an image".
The reason image-style networks are a good fit here is not that weather is a picture; it is that
**convolutions are local and the same everywhere**, which matches local physics and matches the
grader's causality test (a shock must stay near its box on day 1). Non-image families (trees,
graphs, physics-hybrids) are also on the list below, and we will train a baseline of each.

The grader rewards three things that pull in different directions:
1. **Low error (RMSE)**: rewards smooth, average-looking forecasts.
2. **Sharpness (anti-blur, heavy-rain CSI)**: rewards keeping realistic detail and strong rain.
3. **Causality**: rewards local, gradual spreading of a shock and punishes any global mixing.

Every family below is rated against those three.

---

## 1. Classical baselines (no learning or almost none)

- **Persistence**: tomorrow = today.
- **Climatology**: tomorrow = the long-term average for that calendar day.
- **Damped anomaly persistence**: `forecast = clim + a^lead * (today - clim_today)`, one `a` fitted
  per channel and lead.

| | |
|---|---|
| Image model? | No |
| Strengths | Instant, cannot explode, deterministic, easy to explain. Damped persistence already reached mean skill ~0.11 on held-out years (the grader's full-mark level is 0.12). |
| Weaknesses | No movement: a shock never spreads, so it can never pass the "propagates" check. Rain is blurry. |
| Role for us | The floor every other model must beat, a safe fallback model, and a good "prior" to add learned corrections on top of. |

## 2. Gradient-boosted trees per cell (LightGBM)

One tabular model per channel predicts the next-day value of a cell from features of its
neighbourhood (a small patch of the 3 input days), the static maps and the day of year.

| | |
|---|---|
| Image model? | No (tabular) |
| Strengths | Strong on small data, fast to train on CPU, handles rain's skewed distribution well, locality is guaranteed by the patch size. `lightgbm` is available in the judge sandbox. |
| Weaknesses | Hundreds of thousands of rows per day of data; rolling out 10 steps means 10 rounds of feature building (must stay under 5 s). Outputs are piecewise constant and tend to be smooth. No learned motion beyond the patch. Repo scan counts `.fit(` as training, so it is fine. |
| Role | A strong non-neural baseline; also good for the rain head of a hybrid. |

## 3. Local CNN (ResNet-style, no down-sampling)

Stacked 3x3 convolutions with skip connections at full 32 x 36 resolution. The receptive field is
set by depth (for example 6 layers of 3x3 = 6 cells per step).

| | |
|---|---|
| Image model? | Yes |
| Strengths | Locality is exact and controllable, so the causality tests are easy. Few parameters, so it fits our small dataset. Very fast on CPU. Translation-equivariant: the same rule everywhere, which is how physics behaves. |
| Weaknesses | Sees only a few cells per step, so large-scale patterns (monsoon trough, pressure systems) must be built up over steps. Plain MSE training makes it blurry. |
| Role | The main candidate. Simplest network that matches the physics and the grading. |

## 4. U-Net (multi-scale CNN)

Encoder shrinks the grid (32x36 to 16x18 to 8x9), decoder grows it back, with skip connections.

| | |
|---|---|
| Image model? | Yes |
| Strengths | Sees both local detail and large patterns in one step. The standard choice for grid-to-grid prediction. Usually the lowest RMSE among small models. |
| Weaknesses | Deep levels see the **whole grid** in one step, which can leak a shock far away on day 1 (fails "localized" or "no teleport"). Needs care: only 1-2 down-sampling levels, or a capped receptive field. Blurry under MSE. |
| Role | Strong RMSE candidate; must be checked hard against the causality tests. |

## 5. ConvLSTM / ConvGRU (recurrent CNN)

A CNN whose cells keep a hidden memory across time steps.

| | |
|---|---|
| Image model? | Yes (image + time) |
| Strengths | Natural fit for sequences; the hidden state can carry slow signals (soil moisture, ocean heat) that 3 input days do not show. Local like a CNN. |
| Weaknesses | Slower and harder to train, hidden state can drift over 10 steps, harder to explain. Only 3 input days to warm up the memory. |
| Role | Worth one baseline run; promote only if it clearly beats the CNN. |

## 6. Graph neural network (GraphCast-style, small)

Cells are nodes, neighbours are edges; layers pass messages along edges.

| | |
|---|---|
| Image model? | No (graph) |
| Strengths | Locality is explicit (k message-passing rounds = k hops). Can add longer edges or a coarse mesh in a controlled way. Can treat edge cells differently. State-of-the-art at global scale (GraphCast). |
| Weaknesses | On a small regular grid it is roughly a CNN with more code and slower CPU speed. No `torch_geometric` in the sandbox, so we would write message passing by hand. |
| Role | One baseline run to compare; probably not the winner on a 32x36 grid. |

## 7. Vision transformer / windowed transformer (Pangu / FuXi / ClimaX style)

Cut the grid into patches; attention lets patches exchange information.

| | |
|---|---|
| Image model? | Yes (transformer on image patches) |
| Strengths | Learns long-range links. State of the art at global scale with huge data. |
| Weaknesses | Global attention mixes the whole grid in one step, which breaks the causality tests; must use local windows only. Data-hungry: 6,200 samples is far too few for a transformer to shine. Patching blurs small features like cyclones. |
| Role | One small windowed run for completeness; low expectations. |

## 8. Neural operators (FNO / AFNO / SFNO, FourCastNet style)

Mix information in Fourier space (global waves).

| | |
|---|---|
| Image model? | Grid-to-grid, image-like input |
| Strengths | Very fast, good at smooth large-scale flow. |
| Weaknesses | Every Fourier mode touches every cell, so a shock reaches the whole grid instantly: **fails localization and no-teleport by design**. Assumes the grid wraps around at the edges, which is false for a regional box. |
| Role | Ruled out for the main model. Mention in the brief as a considered-and-rejected option. |

## 9. Generative models (diffusion, GAN, VAE; GenCast style)

Learn to sample realistic future states instead of the average.

| | |
|---|---|
| Image model? | Yes (usually a U-Net inside) |
| Strengths | Sharpest outputs, realistic rain extremes, natural ensembles for uncertainty. |
| Weaknesses | Diffusion needs many network passes per step: 10 steps x 20-50 passes may break the 5 s CPU limit. Random by nature: the null-shock test needs a fixed seed and identical noise per call. Samples score worse RMSE than the mean. GANs are unstable to train. |
| Role | Optional stretch: a tiny conditional "sharpener" only for rain, on top of a deterministic model. |

## 10. Physics-hybrid (learned advection + learned residual, NeuralGCM-lite)

Each step: move fields along the (predicted) wind with a semi-Lagrangian advection step, then a
small CNN adds a learned correction (sources, sinks, rain formation).

| | |
|---|---|
| Image model? | Partly (the correction is a CNN) |
| Strengths | Real "physics-aware" behaviour, which the brief and judges value. Shocks spread at the speed of the wind, so the propagation and no-teleport checks are physically grounded. Advection keeps features sharp instead of blurring them. Great story for the demo ("the warm water moves with the wind"). |
| Weaknesses | More code. Advection needs boundary values at the inflow edge. Must be differentiable for training (bilinear sampling is). |
| Role | Strong candidate for the final model and our USP. |

---

## Ensembles

| Type | What it is | Effect on our score |
|---|---|---|
| Multi-model average | Average several families | Lower RMSE, but averaging **removes variance**, so it hurts anti-blur and heavy-rain CSI. Use per channel: average where it helps (t2m, msl), keep a sharp member for rain and wind. |
| Seed ensemble | Same model, different training seeds | Same trade-off as above, smaller. |
| Perturbed initial states | Run one model on slightly changed inputs | Gives uncertainty for the dashboard. Must use a **fixed** set of perturbations so the null-shock test still matches exactly. |
| Stacking / blending | Learn per-channel, per-lead weights between models (for example CNN early, climatology late) | Cheap, usually the best RMSE gain. Weights are fitted on validation years. |
| Mean + variance restore | Use the ensemble mean, then rescale anomalies back to the real spread | Keeps RMSE gain while passing anti-blur. |

All ensembles must fit in 5 s for 10 steps on 4 CPUs, including JSON time (about 0.5 s budget
for JSON). Each member is cheap here, so 3-5 small members are realistic.

## Plan

1. Train one baseline per family with the **same** data split, inputs, loss and evaluation
   (`training/evaluate.py`, which copies the grader's scoring code).
2. Rank on a local leaderboard that reports every grader component, not just RMSE:
   RMSE skill, CSI skill, anti-blur, causality (4 checks x 2 shocks), null test, speed.
3. Go deeper on the top 1-2 families, then build ensembles from the best members.
