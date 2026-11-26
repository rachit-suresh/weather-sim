# Dashboard (Multiverse Map)

Static pre-built bundle of the **Multiverse Map** pixel-art Radio-Garden globe & branch-timeline frontend (`../frontend/`).
Requires no Python in the browser or frontend build step to run; it talks directly to `serve.py` over HTTP (`/health`, `/meta`, `/simulate`).

## Run (static, zero build step)

From the repository root:

```bash
python serve.py --port 8000                  # backend model server (CORS open)
python -m http.server 5500 -d dashboard      # open http://127.0.0.1:5500
```

When opened on port `5500`, the UI defaults its backend target to `http://127.0.0.1:8000` (editable from the header status button).

## Frontend Development (`../frontend`)

```bash
cd frontend
npm ci
npm run dev       # Vite dev server on http://127.0.0.1:5173 (proxies to 127.0.0.1:8000)
npm run build     # Type-checks and emits fresh static assets into ../dashboard
```

## Layout

| Path | What |
|---|---|
| `index.html`, `assets/*` | Built static frontend emitted from `../frontend` |
| `data/demo_windows.json` | 5 real 3-day start states `[3][6][32][36]` from held-out years (inputs only) |
| `../frontend/src/` | React + TypeScript + Vite source (Globe view, Flat pixel grid, Shock brush, Branch timeline, Impact panel) |
