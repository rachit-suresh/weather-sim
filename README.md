# weather-sim: Multiverse of Monsoon digital twin (Team Sage)

Yugaantar DevForge, Track 2. A fast, local, rain-aware digital twin of India's weather: from 3 days of state it
rolls the weather forward day by day (Timeline Prime) and answers "what if" shocks (Multiverse).
Brief: [`BRIEF.pdf`](BRIEF.pdf). Spec (source of truth): [`docs/SPEC.md`](docs/SPEC.md).

## 1. Run it (what the judge runs, fully offline)

```bash
pip install -r requirements.txt        # torch (CPU), numpy, fastapi, uvicorn - all in the judge sandbox
python serve.py --port 8000            # listens on 0.0.0.0; ready in about 3 s
```

- `POST /health` -> `{"status": "ready", "model_name": "cnn:final_cnn"}`; `POST /simulate` follows `API_CONTRACT.md`
  (physical units, nested JSON lists). A 10-day forecast takes about 40-100 ms on 4 CPU cores.
- Everything the server needs is in this repo as plain files (no Git LFS, nothing downloaded): weights in
  `artifacts/models/`, climatology and static maps in `artifacts/climatology.npz`.
- The served model is chosen in `artifacts/active.json`. If it ever fails to load or forecast, a damped-persistence
  fallback answers, so the server always starts and every request gets a valid answer.
- Check it with the official kit: `python -m monsoon_r1.schema_validator http://127.0.0.1:8000`
  (on Windows use `127.0.0.1`, not `localhost`, which adds about 2 s per call).

## 2. Dashboard (Multiverse Map)

```bash
python serve.py --port 8000                     # terminal 1
python -m http.server 5500 -d dashboard         # terminal 2, open http://127.0.0.1:5500
```

Prebuilt static app in `dashboard/`, React + TypeScript source in `frontend/` (`cd frontend && npm ci && npm run build`
rebuilds `dashboard/`). Every map is a live call to the server; nothing is pre-rendered. Shock brush and the judges'
presets, Prime vs Shocked branch timelines, impact lenses (flood risk, wet-bulb heat stress, crop water balance,
wind and solar). Server contract for the frontend: `docs/DASHBOARD_API.md`.

## 3. The submitted model

`artifacts/models/final_cnn.pt`: a local residual CNN (7 layers of 3x3 convolutions, 239k weights) that predicts the
next day's change in anomaly, with a two-part rain head ("does it rain?" + "how much?"), trained on **all
2005-2021 data**. How it works: `docs/HOW_THE_MODELS_WORK.md`.

The recipe was chosen by measurement before this final retrain: 30+ models from 9 families were trained on
2005-2017, compared on 2018-19 and checked on untouched 2020-21, all scored with the organisers' own scoring code
(`training/evaluate.py`). The chosen recipe (`r3_cnn`) scored 67.9 / 70 on 2018-19 and 68.0 / 70 on 2020-21 on the
judge's automatic parts. Full history: `docs/RESULTS_SWEEP_1.md`, `docs/RESULTS_RUN_2.md`, `docs/RESULTS_RUN_3.md`.

## 4. Reproduce the submitted model from scratch

```bash
pip install -r requirements-train.txt
pip install monsoon_r1_kit-1.0.0-py3-none-any.whl    # organisers' kit from their Release page (scoring, validator, scan)
# put india_climate_train.nc next to the repo folder (../india_climate_train.nc) or set TWIN_DATA=/path/to/it

python -m training.build_artifacts                   # artifacts/climatology.npz + stats.json (2005-2021)
python -m training.fit_damped                        # artifacts/damped.json (fallback model)
python -m training.train_nn --family cnn --name final_cnn --rain-head two_part \
       --train-years 2005-2021 --final --threads 8   # artifacts/models/final_cnn.pt (CPU, about 1 hour)
```

The other defaults of `train_nn` are the recipe in `docs/SPEC.md`: square-root rain, 3x weight on cells above
10 mm, spread + spectral losses, unroll curriculum 1-2-4-8 days, seed 0. Training saves its full state every epoch,
so `--resume` continues after an interruption. Then point `artifacts/active.json` at
`{"model": "neural", "checkpoint": "models/final_cnn.pt"}`.

To reproduce the model-selection runs (train 2005-2017, score on held-out years):

```bash
python -m training.run2 --family cnn --prefix r3 "--options=--rain-head two_part"     # = r3_cnn
python -m training.evaluate --checkpoint models/r3_cnn.pt --split test               # 2020-21 check
python -m training.report --prefix r3_                                               # leaderboard table
```

## 5. Checks before every push

```bash
python tests/test_engine.py                          # shapes, validity, zero shock == no shock, speed
python scripts/check_repo.py                         # judge's repo scan on tracked files: must say 15/15
python -m monsoon_r1.schema_validator http://127.0.0.1:8000
```

Repo-scan rules that bite (`docs/GRADING_NOTES.md`): no Python file outside the server path may contain the route
string of the simulate endpoint; no calendar dates or weather-data-source names in `serve.py` or `twin/`; no `.nc`
files and no multi-year `.npy/.npz` arrays in the repo.

## 6. Repository map

| Path | What |
|---|---|
| `serve.py` | Server entry point (FastAPI) |
| `twin/` | Inference package: engine (shock, clamps, fallback), model families, codec, rollout |
| `training/` | Data split, training, post-processing fitters, local grader, reports (never imported by the server) |
| `artifacts/` | Everything the server loads: `active.json`, `models/*.pt`, `climatology.npz`, `stats.json`, `damped.json` |
| `dashboard/`, `frontend/` | Multiverse Map: prebuilt static app and its source |
| `docs/` | Spec, results, model explanations, grading notes, dashboard contract, brief generator (`docs/brief/`) |
| `scripts/`, `tests/` | Repo scan, sweep/run launchers, engine contract tests |
| `runs/leaderboard.csv` | Every evaluation we ran, with all score components |

Licence: code MIT. Data: modified ERA5 (Copernicus Climate Change Service, CC BY 4.0) via the organisers' kit.
