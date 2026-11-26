# weather-sim: Multiverse of Monsoon digital twin

Yugaantar DevForge, Track 2. A fast climate digital twin of India: from 3 days of state it rolls the
weather forward 10 days (Timeline Prime) and answers "what if" shocks (Multiverse).

| Folder | What | Owner |
|---|---|---|
| `serve.py` | The judge's entry point: `python serve.py --port 8000` | models |
| `twin/` | Inference package: engine, model families, codec, clamps (imported by the server) | models / optimisation |
| `training/` | Data loading, training, local grader, artifact builders (never imported by the server) | models |
| `artifacts/` | What the server loads: climatology, stats, fitted baselines, `models/*.pt` (Git LFS) | models |
| `dashboard/` | HTML/JS dashboard + impact module, talks to the server over HTTP | visualisation |
| `docs/` | `MODEL_FAMILIES.md`, `DECISIONS.md`, `GRADING_NOTES.md` | everyone |
| `scripts/` | `check_repo.py` (judge's repo scan) | everyone |
| `tests/` | Engine contract tests | everyone |

## Setup

```bash
python -m venv .venv && . .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt                        # what the server needs
pip install -r requirements-train.txt                  # training + evaluation
pip install path/to/monsoon_r1_kit-1.0.0-py3-none-any.whl   # official kit (scoring, validator, repo scan)
git lfs install && git lfs pull                        # model weights
```

Put `india_climate_train.nc` next to the repo folder (`../india_climate_train.nc`) or set `TWIN_DATA`.
Never commit it.

## Run the server

```bash
python serve.py --port 8000
python -m monsoon_r1.schema_validator http://127.0.0.1:8000      # official contract check
```

On Windows use `127.0.0.1`, not `localhost`: `localhost` tries IPv6 first and adds ~2 s per call
(the server listens on IPv4 `0.0.0.0`). Measured: 2.1 s via `localhost`, 0.05 s via `127.0.0.1`.

The active model is set in `artifacts/active.json` (or `TWIN_MODEL_SPEC=path.json`):

```json
{"model": "damped_anomaly"}
{"model": "neural", "checkpoint": "models/cnn_base.pt"}
{"model": "ensemble", "members": [{"model": "neural", "checkpoint": "models/a.pt"}, {"model": "damped_anomaly"}], "weights": [0.7, 0.3]}
```

## Dashboard

```bash
python serve.py --port 8000                  # terminal 1
python -m http.server 5500 -d dashboard      # terminal 2, then open http://localhost:5500
```

See `dashboard/README.md`.

## Train and evaluate

```bash
python -m training.build_artifacts           # climatology.npz + stats.json
python -m training.fit_damped                # family 1 baseline
python -m training.train_nn --family cnn --name cnn_base
python -m training.evaluate --checkpoint models/cnn_base.pt --split val
```

Every evaluation appends a row to `runs/leaderboard.csv` with every judge score component
(RMSE skill, heavy-rain CSI, anti-blur, shock causality, null test, speed), computed with the
kit's own scoring code on held-out years. Splits: train 2005-2017, val 2018-2019, test 2020-2021.

## Before every push

```bash
python tests/test_engine.py
python scripts/check_repo.py     # must say 15/15
```

Repo scan rules that bite (details in `docs/GRADING_NOTES.md`):
- No Python file outside the server path may contain the route string of the simulate endpoint
  (it would be treated as inference code). The dashboard is JS for this reason.
- No calendar dates or weather-data-source names in `serve.py` or `twin/` (comments included).
- No `.nc` files and no multi-year `.npy/.npz` archives in the repo.

## Model family sweep

Results and lessons: `docs/RESULTS_SWEEP_1.md`. How each model works: `docs/MODEL_FAMILIES.md`,
`docs/PHYSICS_HYBRID.md`. Finish an interrupted sweep: `bash scripts/resume_sweep.sh <python>`.
`train_nn.py` saves full training state every epoch (`runs/ckpt/`), so `--resume` continues after a shutdown.

## First baselines (val 2018-2019, auto-scored parts out of 70)

| Model | events | random | notes |
|---|---|---|---|
| persistence | 21.3 | 19.6 | fails the day-4 gate |
| climatology | 9.7 | 10.0 | ignores shocks |
| damped anomaly | 43.7 | 42.3 | safe fallback |
| CNN, 2-epoch smoke test | 52.0 | 57.1 | pipeline check only |

Licence: code MIT. Data: modified ERA5 (Copernicus, CC BY 4.0) via the organisers' kit.
