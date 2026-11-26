"""Inference server. The judge runs:  python serve.py --port 8000   (offline, 4 CPU, 6 GB, read-only repo).

POST /health    -> {"status": "ready", "model_name": ...}
POST /simulate  -> {"rollout": [steps,6,32,36], "inference_ms": float}   physical units, nested lists
GET  /meta      -> grid, channels, units, model name (used by the dashboard)

The dashboard lives in dashboard/ and talks to this server over HTTP only (CORS is open for that).
Keep UI libraries out of this file and everything it imports.
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import time

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from twin.climatology import load_static
from twin.config import CHANNELS, LAT, LON, MAX_STEPS, UNITS
from twin.engine import Engine

logging.basicConfig(level=logging.WARNING)
ENGINE = Engine()                       # model loads once, at import time

app = FastAPI(title="Multiverse of Monsoon twin")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _arr(x):
    """Nested lists (the contract) or {'b64': ..., 'shape': [...]} (tolerated)."""
    if isinstance(x, dict):
        return np.frombuffer(base64.b64decode(x["b64"]), dtype="<f4").reshape(x["shape"]).astype("f4")
    return np.asarray(x, dtype="f4")


@app.api_route("/health", methods=["GET", "POST"])
def health():
    return {"status": "ready", "model_name": ENGINE.name}


@app.get("/meta")
def meta():
    return {"model_name": ENGINE.name, "channels": list(CHANNELS), "units": list(UNITS),
            "lat": LAT.tolist(), "lon": LON.tolist(), "max_steps": MAX_STEPS,
            "land_sea_mask": np.round(load_static()[0], 3).tolist()}


@app.post("/simulate")
async def simulate(request: Request):
    t0 = time.perf_counter()
    body = await request.json()
    try:
        init = _arr(body["init_states"])
        pert = None if body.get("perturbation") is None else _arr(body["perturbation"])
        out = ENGINE.simulate(init, body["doy_sincos"], int(body.get("steps", 10)), pert)
    except (KeyError, ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=f"bad request: {type(e).__name__}: {e}")
    payload = {"rollout": out.tolist(), "inference_ms": (time.perf_counter() - t0) * 1000.0}
    return Response(content=json.dumps(payload, separators=(",", ":")), media_type="application/json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    a = ap.parse_args()
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
