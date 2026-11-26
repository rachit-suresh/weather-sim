# How the physics hybrid works (family 10)

Code: `twin/nets/physics.py` (`AdvectionHybrid`). About 75k parameters.

## The idea in one sentence

Weather mostly **moves** (a hot air mass, a rain band, a cyclone drifts with the wind) and partly
**changes in place** (the sun heats land, rain forms and stops). So each forecast day we do both
explicitly: first move yesterday's weather along a learned flow, then add learned changes.

```
today's anomaly ──► [1. flow net] ──► displacement map (dy, dx) per cell, per variable, max 4 cells/day
       │                                   │
       └──────────────► [2. advection: look up each cell's "departure point"] ──► moved anomaly
                                                                                     │
inputs + moved anomaly ──► [3. source net] ──► heating / rain formation / decay ──► + ──► tomorrow's anomaly
```

## The three parts

1. **Flow net** (small CNN, 3 layers). Reads the 3 input days, the season and the terrain. For every
   cell and every variable it outputs how far the air moved in one day, as (dy, dx) in grid cells.
   A `tanh` caps it at 4 cells/day (about 450 km/day, a fast but physical wind speed).

2. **Semi-Lagrangian advection** (no learned weights). This is the standard method real weather
   models use. To get tomorrow's value at cell X, ask "where was this air yesterday?", which is
   X minus the displacement, and read yesterday's value there with bilinear interpolation.
   The pattern **moves** instead of being smeared, so features stay sharp.

3. **Source net** (small CNN, 3 layers). Sees the inputs and the moved field, and adds what
   motion cannot explain: daytime heating, moisture turning into rain, anomalies fading back
   toward normal.

Both nets start at zero output, so before training the model is exactly "persistence", and
training only has to learn the departures from that.

## Why it suits this contest

| Grader check | Why the hybrid helps |
|---|---|
| Causality: shock stays local on day 1 | A shock can only move <= 4 cells (plus a few cells of source-net reach) per day |
| Causality: spreads outward later | The flow carries the shock downstream day after day (e.g. a warm Bay of Bengal drifts inland with the monsoon flow) |
| No teleport (< 8 cells/day) | Hard-capped by the flow limit |
| Anti-blur | Moving a pattern does not average it away (unlike a CNN trained on MSE) |
| Brief + desk review ("physics-aware") | Every step has a physical meaning, and we can draw the learned flow arrows on the dashboard |

## Weak points (honest list)

- **Bilinear lookup adds some smoothing** each day (numerical diffusion). Over 10 days a sharp
  feature blurs a bit. Fix if needed: bicubic lookup or a small sharpening term.
- **Inflow at the edges** is still invented: at the domain edge the lookup clamps to the border
  value (`padding_mode="border"`).
- **Surface wind is not the steering wind.** The flow is learned, not taken from u10/v10, so it can
  learn the real steering flow. It is also less directly "physical" than using the wind.
- **Each variable has its own flow.** That is flexible, but less physical than one shared flow.
  It is a design choice we can test.
- **Small model.** The source term may be too weak for rain formation. Width can grow.

## Variants we can test if the family wins

1. One shared flow for all variables, plus a per-variable correction.
2. Flow initialised from the physical wind (u10, v10 converted to cells/day) plus a learned correction.
3. Two half-steps per day (a smoother, more accurate path).
4. A sharper interpolation scheme.
5. A bigger source net, or a CNN or GRU as the source net (combines with family 3 or 5).
