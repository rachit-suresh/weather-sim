# Dashboard

Plain HTML/JS, no build step, no Python. It only talks to the server over HTTP (`/meta`, `/health`,
and the simulate endpoint), so every forecast shown is live (a judge rule: no pre-rendered results).

## Run

```bash
python serve.py --port 8000                  # from the repo root (CORS is open)
python -m http.server 5500 -d dashboard      # then open http://localhost:5500
```

Works with any model the server runs, including the baselines, so you can build the UI before the
final model exists.

## Files

| File | What |
|---|---|
| `index.html`, `style.css` | Layout |
| `app.js` | Server calls, map rendering (canvas), shock editor (drag a box on the map), Prime / Shocked / Difference views |
| `impact.js` | Impact indicators computed from the live rollout. Add yours to `INDICATORS`. |
| `data/demo_windows.json` | 5 real 3-day start states from held-out years (inputs only). Regenerate: `python -m training.export_demo_windows` |

## Server API used

- `GET /meta` -> channels, units, lat, lon, `land_sea_mask` [32][36], model name
- `POST` simulate endpoint: `{init_states [3][6][32][36], doy_sincos [2], steps, perturbation [6][32][36] | null}`
  -> `{rollout [steps][6][32][36], inference_ms}`
- Grid row 0 is the SOUTH edge (lat 6.5). Channel order: t2m K, tp mm/day, u10 m/s, v10 m/s, msl Pa, tcwv kg/m2.

## Ideas / TODO (Stage 2 demo)

- Real basemap (state borders as a static GeoJSON drawn on the canvas; must work offline)
- Wind arrows (u10, v10), animation over days, side-by-side Prime vs Shocked
- Time series at a clicked cell, region summaries (Kerala, Indo-Gangetic plain, ...)
- Impact extension (pick one, go deep): crop stress under a stalled monsoon, heatwave risk,
  flood risk, solar/wind output, ensemble uncertainty
- Custom shock shapes (circle, gaussian blob) and saving shock scenarios
