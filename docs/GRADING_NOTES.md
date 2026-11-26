# Grading notes (read from the kit source, `monsoon_r1` v1.0.0)

Where the kit wiki and the kit code disagree, the code wins.

## Requests
8 blind `/simulate` calls: 5 real windows (2023-2024: monsoon onset, heavy rain, heatwave,
cyclone, winter disturbance), then 3 shocks on one real window: `warm_ocean`, `monsoon_stall`,
`null`. Each call has a 5.0 s limit, no retries. `/health` (POST) is polled up to 120 s first.

## Points
| Part | Points | Rule |
|---|---|---|
| RMSE skill | 30 | `SS = 1 - RMSE/min(RMSE_persist, RMSE_clim)`, lat-weighted, leads 1/4/8, 6 channels. Full at mean SS 0.12. |
| Heavy rain | 15 | CSI at > 10 mm/day vs the better reference, `(CSI-ref)/(1-ref)`. Full at 0.10. Windows with no events are skipped. |
| Anti-blur | 10 | Day-8 spatial variance / real, per channel, averaged over windows. Full at >= 0.4, **zero above 2.5**. |
| Causality | 15 | Per shock: nonzero, localized (>= 60% of day-1 energy within 3 cells of the box), propagates (outside energy grows by days 4-8, < 5% beyond `3 + 8*day` cells), stable (never > 5 std). Not localized -> only "nonzero" counts. Failed null test halves the 15. |
| Repo scan | 15 | 8 anti-cheat + 7 code health. Run `python scripts/check_repo.py`. |
| Brief | 15 | Read by a person. |

Hard gate: beat persistence at day 4, valid output on all 8, localized response to one shock.

## Measured on our data (held-out 2020-2021)
- Persistence wins at day 1; climatology wins at days 4 and 8 for every channel.
- Damped anomaly persistence (no ML) reaches mean SS ~0.11.
- Climatology keeps only 32-58% of real day-8 rain variance (season dependent): blur risk for rain.
- Heavy-rain CSI references: persistence ~0.43 (day 1), climatology ~0.27 (all leads).

## Repo scan traps (from `repo_scan.py`)
- "Inference code" = any `.py` containing `"/simulate"` or `'simulate'` plus everything it imports.
  A Python dashboard calling `/simulate` becomes inference code. That is why the dashboard is JS.
- In inference code: 3-4 date literals = warn (-2), 5+ = critical (-8). Comments count.
- In inference code, these words are critical even in comments: open-meteo, openweathermap,
  weatherapi, cdsapi, weatherbench2, gcp-public-data-arco-era5, imdlib, mosdac, noaa-gfs, ...
- Any `.nc/.grib/.zarr` in the repo: -2. Any `.npy/.npz` with > 400 days of 32x36: critical.
- Training code must live in files the server does not import, and contain e.g. `loss.backward(`.
- Need a weight file > 1 KB (`.pt .onnx .npz ...`), a root README and a frontend file.

## Runtime rules
- `python serve.py --port 8000`, 0.0.0.0, 6 GB RAM, 4 CPU, no GPU, no internet, read-only repo
  (write only to /tmp).
- Libraries: torch (CPU), numpy, scipy, pandas, xarray, scikit-learn, lightgbm, onnxruntime,
  fastapi, flask, uvicorn. Anything else must be vendored.
- Clamp outputs: rain >= 0, t2m in [233.15, 333.15] K, finite only.
