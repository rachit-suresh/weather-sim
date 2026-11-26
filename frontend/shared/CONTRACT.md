# `/simulate` contract — Multiverse Map dashboard

**Owner:** Person 3 (dashboard). **Consumer of:** Person 1/2 `serve.py`.
**Status:** the dashboard ships against this contract. A mock server
(`server/mock-simulate.mjs`) implements it exactly, so the UI is fully
demoable before the model is wired up.

> **To switch from mock to the real model, change one line.**
> Set `VITE_SIMULATE_URL` (see `.env.example`) or edit `TARGET` in
> `src/lib/api.ts`. Nothing else in the UI touches the network.

---

## Grid indexing (non-negotiable)

| property | value |
|---|---|
| shape | `32` rows (lat) × `36` cols (lon) = **1152 cells** |
| spacing | 1.0° |
| row 0 | **37.5°N** (north) — rows run **north → south** |
| row 31 | 6.5°N |
| col 0 | **67.5°E** (west) — cols run **west → east** |
| col 35 | 102.5°E |
| flat index | `i = row * 36 + col` |

`lat[r] = 37.5 - r`, `lon[c] = 67.5 + c`.
The canonical machine-readable version is **`shared/grid.json`** (also served at
`GET /meta`). It carries the landmask, orography, regions and reference cities.
Both the server and the browser import it, so they cannot disagree.

Landmask values: `0` ocean · `1` land outside India · `2` India land.

## Variables

| key | meaning | unit |
|---|---|---|
| `t2m` | 2 m temperature | K |
| `tp` | total precipitation (daily rate) | mm/day |
| `tcwv` | total column water vapour | kg/m² |
| `u10` | 10 m zonal wind (+ = eastward) | m/s |
| `v10` | 10 m meridional wind (+ = northward) | m/s |
| `msl` | mean sea-level pressure | hPa |

`tp` is a **daily rate**, not an accumulation. If `serve.py` emits metres of
accumulated precip, convert in the adapter (`mm/day = m * 1000 / step_days`) —
`src/lib/api.ts` has a hook for exactly this.

---

## `POST /simulate`

### Request

```jsonc
{
  "horizon_days": 10,                  // 1..10
  "variables": ["t2m","tp","tcwv","u10","v10","msl"],
  "seed": 7,                           // reproducibility
  "include_prime": true,               // also return the unperturbed control
  "perturbations": [
    // form A — explicit painted mask (what the Shock Painter sends)
    { "field": "t2m", "delta": 2.0,
      "mask": [[22,21,1.0],[22,22,0.83],[23,21,0.61]] },   // [row, col, weight 0..1]

    // form B — analytic stamp (what the region presets send)
    { "field": "u10", "delta": -3.0,
      "center_latlon": [15.5, 70.5],   // OR "center": [row, col]
      "radius_cells": 3.0,
      "falloff": "gaussian" }          // gaussian | linear | hard
  ]
}
```

Rules
* `perturbations: []` ⇒ the branched run **is** the prime run (all deltas 0).
* `delta` is added to the **initial condition** of `field`, in that field's unit.
* `mask` weights are clamped to `[0,1]`; max 4096 entries per perturbation.
* Max 24 perturbations; `|delta| <= 50`.
* Perturbations are applied at day 0 and then evolve freely.

### Response `200`

```jsonc
{
  "run_id": "sur_0007_k3f9",
  "mode": "surrogate",                 // "surrogate" | "model"  <-- UI badges this
  "engine": "monsoon-anomaly-surrogate/1.1",
  "grid": { "nlat":32, "nlon":36, "order":"lat_desc_lon_asc",
            "lat":[...32], "lon":[...36] },
  "days": 10,
  "variables": ["t2m","tp","tcwv","u10","v10","msl"],
  "units": { "t2m":"K", "tp":"mm/day", ... },
  "layout": "day-major: index = day*nlat*nlon + row*nlon + col",
  "prime":    { "t2m":[...36864 floats], "tp":[...], ... },   // null if include_prime=false
  "branched": { "t2m":[...36864 floats], "tp":[...], ... },
  "applied_perturbations": 1,
  "meta": { "latency_ms": 41.2, "seed": 7, "cells": 1152, "notice": "..." }
}
```

* Each array is **flat, day-major**: `value(day, row, col) = arr[day*1152 + row*36 + col]`.
* Length per variable = `days * 1152` (10 days ⇒ 36 864 numbers).
* `mode` is **required**. The dashboard shows a permanent, unmissable
  `SURROGATE` badge whenever `mode !== "model"`, so surrogate output can never
  be mistaken for model output in front of judges.

### Errors

`400` malformed JSON · `422` unknown variable / field, `delta` out of range,
too many perturbations · `500` engine failure. Body: `{ "error": "message" }`.

## `GET /health` → `{ ok, mode, uptime_s, runs, engine }`
## `GET /meta` → grid + landmask + orography + regions + cities + units

---

## If `serve.py` can only return ONE rollout

That's fine — the adapter handles it. Leave `include_prime` unsupported and the
dashboard fires a second call with `perturbations: []`, caches it as the control
timeline, and reuses it for every subsequent shock. See `ensurePrime()` in
`src/lib/api.ts`. Only the branched call is re-issued when the judge repaints.

## Surrogate physics (what the mock actually computes)

Not a weather model. A deterministic anomaly-propagation surrogate:

1. **Prime background** — July monsoon climatology per cell (northwest heat max,
   Ghats/Bay/Cherrapunji rain maxima, Tibetan lapse rate, monsoon-trough low).
2. **Advection** — semi-Lagrangian backtrace with **bicubic** interpolation,
   8 substeps/day (CFL-safe), driven by a low-level steering flow.
3. **Diffusion + decay** — Laplacian, with e-folding times that are longer over
   ocean (SST anomalies persist) than land.
4. **Couplings** — Clausius–Clapeyron (7 %/K), moisture convergence
   (`Q·(−∇·V)·86400`, SI), orographic lift, latent-heating pressure fall,
   geostrophic adjustment, cloud/evaporative cooling.
5. **Mass-field adjustment** — a damped 2-D wave equation on the MSL anomaly
   (c ≈ 12 m/s). This is what lets a Bay of Bengal shock influence the Western
   Ghats even though the low-level jet flows the *other* way.
6. **Sponge layer** — outer 3 cells relaxed toward zero (Davies-style), so
   nothing piles up against the domain edge.

The `dP → dMSL → wind → convergence → dP` loop gain is **≈0.4/day (< 1)**, so
rollouts decay rather than run away. Verified: zero saturated cells across all
presets and all 10 days.
