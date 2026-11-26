/* ═══════════════════════════════════════════════════════════════════════════
   BACKEND API ADAPTER — implements the authoritative serve.py HTTP contract:
     - GET|POST /health -> { status: "ready", model_name }
     - GET /meta        -> { model_name, channels, units, lat, lon, max_steps, land_sea_mask }
     - POST /simulate   -> { init_states[3][6][32][36], doy_sincos[2], steps, perturbation }
                        -> { rollout[steps][6][32][36], inference_ms }
   ═══════════════════════════════════════════════════════════════════════════ */
import {
  CELLS, GRID, NR, NC, VAR_INDEX, VAR_KEYS, uiDeltaToBackend, type VarKey,
} from "./grid";
import type { DemoWindow } from "./windows";

const STORAGE_KEY = "mm_backend_url";
const ENV_DEFAULT = ((import.meta.env.VITE_SIMULATE_URL as string | undefined) ?? "").replace(/\/$/, "");

function defaultBackendUrl(): string {
  if (ENV_DEFAULT) return ENV_DEFAULT;
  if (typeof window !== "undefined") {
    const port = window.location.port;
    // Vite dev (:5173) and preview (:4173) proxy API routes on same origin.
    if (port === "5173" || port === "4173") return "";
  }
  // Static serving (e.g. `python -m http.server 5500 -d dashboard`) targets serve.py on :8000.
  return "http://127.0.0.1:8000";
}

export function getSavedBackendUrl(): string {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved !== null) return saved.replace(/\/$/, "");
  } catch { /* ignore */ }
  return defaultBackendUrl();
}

export function saveBackendUrl(next: string): string {
  const clean = next.trim().replace(/\/$/, "");
  try {
    if (clean === "") localStorage.removeItem(STORAGE_KEY);
    else localStorage.setItem(STORAGE_KEY, clean);
  } catch { /* ignore */ }
  return clean;
}

const endpoint = (baseUrl: string, path: string) => `${baseUrl.replace(/\/$/, "")}${path}`;

export interface HealthResponse {
  status: string;
  model_name: string;
}

export interface MetaResponse {
  model_name: string;
  channels: VarKey[];
  units: string[];
  lat: number[];
  lon: number[];
  max_steps: number;
  land_sea_mask: number[][];
}

export interface PerturbationSpec {
  field: VarKey;
  /** Value in UI display units (°C/K, mm/day, m/s, hPa, kg/m²) or % when valueMode === "pct". */
  value: number;
  valueMode: "abs" | "pct";
  /** [row, col, weight 0..1] triples in backend coordinates (row 0 = 6.5°N). */
  mask?: [number, number, number][];
  center?: [number, number];
  radius_cells?: number;
  falloff?: "gaussian" | "linear" | "hard";
  label?: string;
}

export interface Timeline {
  /** var -> flat day-major array, length days*CELLS (row 0 = 6.5°N south) */
  prime: Record<VarKey, Float32Array>;
  branched: Record<VarKey, Float32Array>;
  days: number;
  variables: VarKey[];
  modelName: string;
  windowName: string;
  windowLabel: string;
  dayOfYear: number;
  runId: string;
  latencyMs: number;
  inferenceMs: number;
  isSurrogateMock: boolean;
}

export class SimulateError extends Error {
  constructor(message: string, readonly status?: number) { super(message); }
}

function validateTensor(arr: unknown, dims: number[], name: string): string | null {
  if (!Array.isArray(arr) || arr.length !== dims[0]) {
    return `${name} expected length ${dims[0]}, got ${Array.isArray(arr) ? arr.length : typeof arr}`;
  }
  if (dims.length === 1) {
    for (let i = 0; i < dims[0]; i++) {
      if (typeof arr[i] !== "number" || !Number.isFinite(arr[i])) {
        return `${name}[${i}] must be a finite number`;
      }
    }
    return null;
  }
  const rest = dims.slice(1);
  for (let i = 0; i < dims[0]; i++) {
    const err = validateTensor(arr[i], rest, `${name}[${i}]`);
    if (err) return err;
  }
  return null;
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(url, init);
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    throw new SimulateError(`Cannot reach backend at ${url || "same-origin proxy"}.`);
  }
  if (!res.ok) {
    let detail = `HTTP ${res.status} ${res.statusText}`;
    try {
      const j = await res.json();
      if (j?.error) detail = `HTTP ${res.status}: ${j.error}`;
      else if (j?.detail) detail = `HTTP ${res.status}: ${typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail)}`;
    } catch { /* ignore */ }
    throw new SimulateError(detail, res.status);
  }
  return res.json() as Promise<T>;
}

export async function checkHealth(baseUrl: string, signal?: AbortSignal): Promise<HealthResponse> {
  const data = await fetchJson<HealthResponse>(endpoint(baseUrl, "/health"), { method: "GET", signal });
  if (!data || typeof data.status !== "string" || typeof data.model_name !== "string") {
    throw new SimulateError("Invalid /health response shape");
  }
  return data;
}

export async function fetchMeta(baseUrl: string, signal?: AbortSignal): Promise<MetaResponse> {
  const meta = await fetchJson<MetaResponse>(endpoint(baseUrl, "/meta"), { method: "GET", signal });
  if (!meta || typeof meta.model_name !== "string") {
    throw new SimulateError("Invalid /meta: missing model_name");
  }
  if (!Array.isArray(meta.channels) || meta.channels.length !== 6 ||
      !VAR_KEYS.every((k, i) => meta.channels[i] === k)) {
    throw new SimulateError(
      `Invalid /meta channels: expected [${VAR_KEYS.join(",")}], got [${String(meta.channels)}]`,
    );
  }
  if (!Array.isArray(meta.units) || meta.units.length !== 6) {
    throw new SimulateError("Invalid /meta: units must have 6 entries");
  }
  if (!Array.isArray(meta.lat) || meta.lat.length !== NR ||
      Math.abs(meta.lat[0] - 6.5) > 0.1 || Math.abs(meta.lat[NR - 1] - 37.5) > 0.1) {
    throw new SimulateError(`Invalid /meta lat: expected 32 rows from 6.5 to 37.5`);
  }
  if (!Array.isArray(meta.lon) || meta.lon.length !== NC ||
      Math.abs(meta.lon[0] - 65.5) > 0.1 || Math.abs(meta.lon[NC - 1] - 100.5) > 0.1) {
    throw new SimulateError(`Invalid /meta lon: expected 36 cols from 65.5 to 100.5`);
  }
  const maskErr = validateTensor(meta.land_sea_mask, [NR, NC], "land_sea_mask");
  if (maskErr) throw new SimulateError(`Invalid /meta ${maskErr}`);
  if (typeof meta.max_steps !== "number" || !Number.isFinite(meta.max_steps) || meta.max_steps < 1) {
    throw new SimulateError("Invalid /meta max_steps");
  }
  return { ...meta, max_steps: Math.min(14, Math.floor(meta.max_steps)) };
}

/**
 * Converts UI PerturbationSpecs into a single [6][32][36] physical-unit delta tensor.
 * Percentage shocks use `init_states[2]` (latest historical input day).
 */
export function buildPerturbationTensor(
  specs: PerturbationSpec[],
  latestState: number[][][],
): { tensor: number[][][] | null; affectedCells: number } {
  if (!specs.length) return { tensor: null, affectedCells: 0 };

  const tensor: number[][][] = Array.from({ length: 6 }, () =>
    Array.from({ length: NR }, () => new Array<number>(NC).fill(0)),
  );
  const touched = new Set<number>();

  for (const spec of specs) {
    const ch = VAR_INDEX[spec.field];
    const applyCell = (r: number, c: number, w: number) => {
      if (r < 0 || r >= NR || c < 0 || c >= NC) return;
      const weight = Math.max(0, Math.min(1, w));
      if (weight <= 0) return;
      const delta = spec.valueMode === "pct"
        ? (latestState[ch][r][c] * spec.value) / 100
        : uiDeltaToBackend(spec.field, spec.value);
      const added = delta * weight;
      if (Math.abs(added) > 0) {
        tensor[ch][r][c] += added;
        touched.add(r * NC + c);
      }
    };

    if (spec.mask && spec.mask.length) {
      for (const [r, c, w] of spec.mask) applyCell(r, c, w);
    } else if (spec.center) {
      const [cr, cc] = spec.center;
      const rad = Math.max(0.4, spec.radius_cells ?? 2.5);
      const falloff = spec.falloff ?? "gaussian";
      const R = Math.ceil(rad * 2.5);
      for (let r = Math.max(0, Math.floor(cr - R)); r <= Math.min(NR - 1, Math.ceil(cr + R)); r++) {
        for (let c = Math.max(0, Math.floor(cc - R)); c <= Math.min(NC - 1, Math.ceil(cc + R)); c++) {
          const d = Math.hypot(r - cr, c - cc);
          let w = 0;
          if (falloff === "hard") w = d <= rad ? 1 : 0;
          else if (falloff === "linear") w = Math.max(0, Math.min(1, 1 - d / rad));
          else w = Math.exp(-0.5 * (d / (rad / 1.6)) ** 2);
          if (w > 0.004) applyCell(r, c, w);
        }
      }
    }
  }

  let hasNonZero = false;
  for (let ch = 0; ch < 6 && !hasNonZero; ch++) {
    for (let r = 0; r < NR && !hasNonZero; r++) {
      for (let c = 0; c < NC; c++) {
        if (Math.abs(tensor[ch][r][c]) > 1e-12) { hasNonZero = true; break; }
      }
    }
  }

  return { tensor: hasNonZero ? tensor : null, affectedCells: touched.size };
}

function packRollout(rollout: number[][][][], steps: number): Record<VarKey, Float32Array> {
  const out = {} as Record<VarKey, Float32Array>;
  for (const v of VAR_KEYS) out[v] = new Float32Array(steps * CELLS);

  for (let s = 0; s < steps; s++) {
    const stepBase = s * CELLS;
    for (let ch = 0; ch < 6; ch++) {
      const target = out[VAR_KEYS[ch]];
      const rows = rollout[s][ch];
      for (let r = 0; r < NR; r++) {
        const rowBase = stepBase + r * NC;
        const cols = rows[r];
        for (let c = 0; c < NC; c++) {
          target[rowBase + c] = cols[c];
        }
      }
    }
  }
  return out;
}

interface RawSimulateResponse {
  rollout: number[][][][];
  inference_ms: number;
}

let primeCache: {
  key: string;
  fields: Record<VarKey, Float32Array>;
  inferenceMs: number;
} | null = null;

export function clearPrimeCache() {
  primeCache = null;
}

let runCounter = 1;

async function callSimulateEndpoint(
  baseUrl: string,
  window: DemoWindow,
  steps: number,
  perturbation: number[][][] | null,
  signal?: AbortSignal,
): Promise<{ fields: Record<VarKey, Float32Array>; inferenceMs: number; latencyMs: number }> {
  const cleanSteps = Math.min(14, Math.max(1, Math.round(steps)));

  const initErr = validateTensor(window.init_states, [3, 6, NR, NC], "init_states");
  if (initErr) throw new SimulateError(`Client validation failed: ${initErr}`);
  const doyErr = validateTensor(window.doy_sincos, [2], "doy_sincos");
  if (doyErr) throw new SimulateError(`Client validation failed: ${doyErr}`);
  if (perturbation !== null) {
    const pertErr = validateTensor(perturbation, [6, NR, NC], "perturbation");
    if (pertErr) throw new SimulateError(`Client validation failed: ${pertErr}`);
  }

  const t0 = performance.now();
  const raw = await fetchJson<RawSimulateResponse>(endpoint(baseUrl, "/simulate"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      init_states: window.init_states,
      doy_sincos: window.doy_sincos,
      steps: cleanSteps,
      perturbation,
    }),
    signal,
  });
  const latencyMs = performance.now() - t0;

  const outErr = validateTensor(raw?.rollout, [cleanSteps, 6, NR, NC], "rollout");
  if (outErr) throw new SimulateError(`Invalid /simulate response: ${outErr}`);

  return {
    fields: packRollout(raw.rollout, cleanSteps),
    inferenceMs: typeof raw.inference_ms === "number" ? raw.inference_ms : latencyMs,
    latencyMs,
  };
}

export async function runBaseline(
  params: { baseUrl: string; window: DemoWindow; steps: number; modelName: string },
  signal?: AbortSignal,
): Promise<Timeline> {
  const { baseUrl, window, steps, modelName } = params;
  const cleanSteps = Math.min(14, Math.max(1, Math.round(steps)));
  const cacheKey = `${baseUrl}|${window.name}|${cleanSteps}`;

  const res = await callSimulateEndpoint(baseUrl, window, cleanSteps, null, signal);
  primeCache = { key: cacheKey, fields: res.fields, inferenceMs: res.inferenceMs };

  return {
    prime: res.fields,
    branched: res.fields,
    days: cleanSteps,
    variables: VAR_KEYS,
    modelName,
    windowName: window.name,
    windowLabel: window.label,
    dayOfYear: window.day_of_year,
    runId: `run_${String(runCounter++).padStart(3, "0")}`,
    latencyMs: res.latencyMs,
    inferenceMs: res.inferenceMs,
    isSurrogateMock: modelName.startsWith("surrogate:"),
  };
}

export async function runBranch(
  params: {
    baseUrl: string;
    window: DemoWindow;
    steps: number;
    modelName: string;
    specs: PerturbationSpec[];
  },
  signal?: AbortSignal,
): Promise<Timeline> {
  const { baseUrl, window, steps, modelName, specs } = params;
  const cleanSteps = Math.min(14, Math.max(1, Math.round(steps)));
  const cacheKey = `${baseUrl}|${window.name}|${cleanSteps}`;

  let primeFields: Record<VarKey, Float32Array>;
  if (primeCache?.key === cacheKey) {
    primeFields = primeCache.fields;
  } else {
    const base = await callSimulateEndpoint(baseUrl, window, cleanSteps, null, signal);
    primeCache = { key: cacheKey, fields: base.fields, inferenceMs: base.inferenceMs };
    primeFields = base.fields;
  }

  const { tensor } = buildPerturbationTensor(specs, window.init_states[2]);
  const branched = await callSimulateEndpoint(baseUrl, window, cleanSteps, tensor, signal);

  return {
    prime: primeFields,
    branched: branched.fields,
    days: cleanSteps,
    variables: VAR_KEYS,
    modelName,
    windowName: window.name,
    windowLabel: window.label,
    dayOfYear: window.day_of_year,
    runId: `run_${String(runCounter++).padStart(3, "0")}`,
    latencyMs: branched.latencyMs,
    inferenceMs: branched.inferenceMs,
    isSurrogateMock: modelName.startsWith("surrogate:"),
  };
}

/** Read one cell's value out of a flat day-major field. */
export const readCell = (f: Float32Array, day: number, i: number) => f[day * CELLS + i];

/** Extract a single cell's multi-day series. */
export function cellSeries(f: Float32Array, i: number, days: number): number[] {
  return Array.from({ length: days }, (_, d) => f[d * CELLS + i]);
}

/** Extract one day's 1152-cell frame as a subarray view (no copy). */
export function frame(f: Float32Array, day: number): Float32Array {
  return f.subarray(day * CELLS, (day + 1) * CELLS);
}

/** Branched − Prime for one day (in raw backend physical units). */
export function diffFrame(t: Timeline, v: VarKey, day: number): Float32Array {
  const b = frame(t.branched[v], day);
  const p = frame(t.prime[v], day);
  const out = new Float32Array(CELLS);
  for (let i = 0; i < CELLS; i++) out[i] = b[i] - p[i];
  return out;
}

export { NR, NC, GRID };
