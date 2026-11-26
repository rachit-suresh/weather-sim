# Backend contract: what the Multiverse Map dashboard sends and expects

**Source of truth:** `serve.py` + `twin/` at the repo root, and the organisers' `API_CONTRACT.md`.
This file describes how `src/lib/api.ts` uses that contract. If they ever disagree, `serve.py` wins
and this file plus `api.ts` must be fixed. There is no mock server: every map is a live call to `serve.py`.

## Grid indexing (non-negotiable)

| property | value |
|---|---|
| shape | `32` rows (lat) × `36` cols (lon) = **1152 cells**, 1.0° spacing |
| row 0 | **6.5°N (south)**, rows run **south → north**, row 31 = 37.5°N |
| col 0 | **65.5°E (west)**, cols run **west → east**, col 35 = 100.5°E |
| cell centre | `lat[r] = 6.5 + r`, `lon[c] = 65.5 + c` |
| flat index | `i = row * 36 + col` |

When drawing, flip rows so north is up. `shared/grid.json` (`order: "lat_asc_lon_asc"`) holds the same
grid plus UI-only extras: `mask` (`0` ocean · `1` land outside India · `2` India land), elevation,
regions and cities. The land/sea fraction used for display comes from `GET /meta` when the backend is up.

## Channels (fixed order, physical units on the wire)

| index | key | meaning | unit on the wire | UI display |
|---|---|---|---|---|
| 0 | `t2m` | 2 m temperature | K | °C (−273.15) |
| 1 | `tp` | precipitation, daily rate (always ≥ 0) | mm/day | mm/day |
| 2 | `u10` | 10 m wind, + = toward east | m/s | m/s |
| 3 | `v10` | 10 m wind, + = toward north | m/s | m/s |
| 4 | `msl` | mean sea-level pressure | **Pa** | hPa (÷100) |
| 5 | `tcwv` | total column water vapour | kg/m² | kg/m² |

Unit conversions happen only for display (`display()` in `src/lib/grid.ts`) and when turning a
user-entered hPa shock into Pa (`uiDeltaToBackend()`). Requests and stored rollouts are raw backend units.

## `GET` / `POST /health`

`200 {"status": "ready", "model_name": "unet:p2_unet_d4st"}`. The header badge shows this `model_name`
with a green dot while the backend answers, and a red `OFFLINE` badge when it does not.

## `GET /meta`

```json
{ "model_name": "...", "channels": ["t2m","tp","u10","v10","msl","tcwv"],
  "units": ["K","mm/day","m/s","m/s","Pa","kg/m2"],
  "lat": [6.5, ..., 37.5], "lon": [65.5, ..., 100.5], "max_steps": 14,
  "land_sea_mask": [[...36] x 32] }
```
`fetchMeta()` rejects the response if the channel order, grid or mask shape differ from the above.

## `POST /simulate`

Request:
```jsonc
{
  "init_states": [[[[...36] x 32] x 6] x 3],   // 3 consecutive days, OLDEST first, physical units
  "doy_sincos": [0.74, -0.67],                 // [sin, cos](2*pi*day_of_year/365) of the LAST input day
  "steps": 10,                                 // 1..14
  "perturbation": null                         // or [6][32][36]: a delta ADDED to init_states[2]
}
```
Response `200`:
```json
{"rollout": [[[[...36] x 32] x 6] x steps], "inference_ms": 42.1}
```
`rollout[k]` is day `k+1` after the last input day. Bad input returns `400 {"detail": "bad request: ..."}`.

### How the dashboard uses it

* **Prime** = one call with `perturbation: null`; cached per (backend URL, window, steps).
* **Branched** = one call with the summed perturbation tensor of every armed shock.
  Armed shocks always send a full `[6][32][36]` tensor, even an all-zero one.
* Shocks are deltas, never new values. A percent shock becomes
  `delta = init_states[2][ch][r][c] * percent / 100` in the browser.
* Painted shocks carry per-cell weights 0..1 (brush falloff); region presets are gaussian stamps.
* **Judge presets** (exact evaluation definitions; a box covers every cell whose centre is inside it):
  * `Warm Bay of Bengal (Judge)`: `t2m` +3 K and `tcwv` +15 % over 8–20°N, 82–94°E
  * `Monsoon Stall (Judge)`: `u10` and `v10` −40 % over 8–20°N, 65–75°E
  * `Zero Shock (Sanity Check)`: all-zero tensor. The server guarantees zero shock == null shock, so
    the branch must equal Prime exactly; the status toast reports `max |Δ| = 0`.
* The backend is deterministic: the same request always returns the same rollout, so
  Branched − Prime is pure shock effect.

## Inputs and real coupling

* Start states come only from `data/demo_windows.json` (real 3-day windows from held-out years, inputs
  only, never forecasts; regenerate with `python -m training.export_demo_windows`).
* If that file is missing or invalid, or the backend is unreachable, the UI shows the banner
  **"Backend offline or demo_windows.json missing"** and renders no fields. There is no synthetic or
  pre-rendered fallback anywhere.

## Backend URL

* `npm run dev` (:5173) / `npm run preview` (:4173): same-origin proxy to `SIMULATE_TARGET`
  (shell environment variable, default `http://127.0.0.1:8000`).
* Static build served from `dashboard/` (e.g. `python -m http.server 5500 -d dashboard`):
  calls `http://127.0.0.1:8000` directly (CORS is open on the server).
* `VITE_SIMULATE_URL` at build time, or the URL field in the header (saved in localStorage), overrides both.
