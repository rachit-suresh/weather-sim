# Dashboard <-> server contract

For whoever builds `dashboard/`. The server is `serve.py` (owned by the models side). The dashboard only talks
to it over HTTP. If you need something the server does not offer, ask for a spec change first (see the rules
at the bottom); do not edit `serve.py` or `twin/` yourself.

## Running

```bash
python serve.py --port 8000                  # repo root
python -m http.server 5500 -d dashboard      # open http://127.0.0.1:5500
```
On Windows use `127.0.0.1`, not `localhost` (localhost adds ~2 s per call). CORS is open (`*`).

## Grid and units (fixed, never changes)

- Every state is `[6][32][36]`: channel, latitude row, longitude column.
- Row 0 = **south** edge (lat 6.5), row 31 = north (lat 37.5). Column 0 = west (lon 65.5), column 35 = east (lon 100.5).
  Cell `[i][j]` is centred at lat `6.5 + i`, lon `65.5 + j` and covers 1 x 1 degree. When drawing, flip rows so north is up.
- Channel order and units:

| index | name | unit | note |
|---|---|---|---|
| 0 | t2m | K | subtract 273.15 for C |
| 1 | tp | mm/day | always >= 0 |
| 2 | u10 | m/s | + = wind blowing toward east |
| 3 | v10 | m/s | + = wind blowing toward north |
| 4 | msl | Pa | divide by 100 for hPa |
| 5 | tcwv | kg/m2 | = mm of water if all vapour fell as rain |

All numbers on the wire are physical units. Wind speed = `sqrt(u10^2 + v10^2)`, direction from the two components.

## Endpoints

### `GET` or `POST /health`
`200 {"status": "ready", "model_name": "cnn:sweep_cnn"}`

### `GET /meta`
```json
{
  "model_name": "cnn:sweep_cnn",
  "channels": ["t2m", "tp", "u10", "v10", "msl", "tcwv"],
  "units": ["K", "mm/day", "m/s", "m/s", "Pa", "kg/m2"],
  "lat": [6.5, ..., 37.5],          // 32 values
  "lon": [65.5, ..., 100.5],        // 36 values
  "max_steps": 14,
  "land_sea_mask": [[...36], ...32] // 1 = land, 0 = sea, fractions on coasts
}
```

### `POST /simulate`
Request:
```json
{
  "init_states": [[[[...36]...32]...6]...3],   // 3 consecutive days, OLDEST first, physical units
  "doy_sincos": [0.42, -0.91],                 // [sin, cos](2*pi*day_of_year/365) of the LAST input day
  "steps": 10,                                 // 1..14
  "perturbation": null                         // or [6][32][36]: a delta ADDED to init_states[2]
}
```
Response `200`:
```json
{"rollout": [[[[...36]...32]...6]...steps], "inference_ms": 42.1}
```
`rollout[k]` = day k+1 after the last input day. Bad input -> `400 {"detail": "bad request: ..."}`.

Facts you can rely on:
- **Deterministic:** the same request always gives the same answer. A zero perturbation gives exactly the
  same answer as `null`. Difference maps (shocked - prime) are therefore pure shock effect.
- **Shock = delta, not a new value.** "+3 K over the Bay of Bengal" = put 3.0 in channel 0 for those cells and 0
  everywhere else. A percentage shock must be turned into a delta on the dashboard side
  (`delta = last_day_value * percent / 100`), as `app.js` does.
- The server clamps wild inputs (e.g. negative rain, -100 C) into physical ranges instead of failing.
- Timing: about 50-150 ms per 10-day rollout; the payload is ~1-3 MB of JSON each way.
- Demo start states: `dashboard/data/demo_windows.json` (5 real windows from 2020: name, label, doy,
  doy_sincos, init_states). Regenerate with `python -m training.export_demo_windows`.

### Planned (not built yet; needs a spec change first)
- Ensemble spread / uncertainty map per day.
- Physics-hybrid flow arrows (how far air moved per day).
- Impact indicators computed server-side (only if they become part of the brief's impact module).

## DO NOTs

1. **Do not write any Python file that calls the simulate endpoint** (no Streamlit/Dash/Flask dashboard, no
   Python test scripts with the route string in them). The judge's repo scan treats any `.py` file containing
   that route as server code: importing plotly/streamlit there costs points, and dates in it cost more.
   The dashboard stays HTML/CSS/JS.
2. **Do not show pre-rendered or saved forecasts.** Every map must come from a live call to our server
   (judge rule "real coupling"). Saved INPUT windows are fine; saved OUTPUTS are not.
3. **Do not edit `serve.py`, `twin/`, `training/` or `artifacts/`.** Ask for an endpoint; it goes into the spec first.
4. **Do not change channel order, units or the grid orientation** anywhere in the dashboard code.
5. **Do not depend on the internet at demo time.** No CDN scripts, no online map tiles, no web fonts. Vendor any
   library into `dashboard/vendor/`, and ship borders (India/states GeoJSON) as a local file.
6. **Do not commit build output or `node_modules/`.** If you use a bundler, commit the source and a built
   `dashboard/dist/` only if needed, and keep it small.
7. **Do not commit climate data files** (`.nc`, `.grib`, `.zarr`, `.npy`/`.npz` with many days): the repo scan
   penalises them. Small JSON inputs like `demo_windows.json` are fine.
8. **Do not send more than 14 steps** or arrays of the wrong shape; check shapes before sending.
9. **Do not hardcode the server URL** to localhost only; keep the editable field (judges may run it differently).
10. **Do not put API keys, secrets or personal data** in the dashboard.

## Rules of the repo (spec-driven)
- The spec is the source of truth. A change goes into the spec first, then into code.
- Before pushing: `python scripts/check_repo.py` must print 15/15.
