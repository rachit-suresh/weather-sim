import { NC, NR } from "./grid";

export interface DemoWindow {
  name: string;
  label: string;
  day_of_year: number;
  doy_sincos: [number, number];
  /** [3 historical consecutive days][6 channels][32 lat rows][36 lon cols], physical units */
  init_states: number[][][][];
}

/** Thrown when no real input windows can be loaded. There is deliberately NO synthetic fallback:
 *  every map must come from real inputs sent live to the backend ("real coupling" rule). */
export class DemoWindowsError extends Error {}

function isValidState3x6x32x36(states: unknown): states is number[][][][] {
  if (!Array.isArray(states) || states.length !== 3) return false;
  for (let d = 0; d < 3; d++) {
    const s = states[d];
    if (!Array.isArray(s) || s.length !== 6) return false;
    for (let ch = 0; ch < 6; ch++) {
      const rows = s[ch];
      if (!Array.isArray(rows) || rows.length !== NR) return false;
      for (let r = 0; r < NR; r++) {
        const cols = rows[r];
        if (!Array.isArray(cols) || cols.length !== NC) return false;
        for (let c = 0; c < NC; c++) {
          if (typeof cols[c] !== "number" || !Number.isFinite(cols[c])) return false;
        }
      }
    }
  }
  return true;
}

export function doyToSinCos(doy: number): [number, number] {
  const rad = (2 * Math.PI * doy) / 365;
  return [+Math.sin(rad).toFixed(6), +Math.cos(rad).toFixed(6)];
}

/**
 * Loads the real historical input windows from `data/demo_windows.json` (inputs only, never forecasts).
 * Throws DemoWindowsError if the file is missing, unreadable or holds no valid [3][6][32][36] window.
 */
export async function loadDemoWindows(signal?: AbortSignal): Promise<DemoWindow[]> {
  let lastProblem = "not found";
  for (const path of ["./data/demo_windows.json", "/data/demo_windows.json"]) {
    try {
      const res = await fetch(path, { signal });
      if (!res.ok) { lastProblem = `HTTP ${res.status}`; continue; }
      const raw = await res.json();
      const arr = Array.isArray(raw) ? raw : Array.isArray(raw?.windows) ? raw.windows : [];
      const valid: DemoWindow[] = [];
      for (const item of arr) {
        if (!item || typeof item.name !== "string" || !isValidState3x6x32x36(item.init_states)) continue;
        const doyRaw = Number(item.day_of_year ?? item.doy);
        const hasSincos = Array.isArray(item.doy_sincos) && item.doy_sincos.length === 2 &&
          item.doy_sincos.every((x: unknown) => Number.isFinite(Number(x)));
        if (!hasSincos && !Number.isFinite(doyRaw)) continue;          // no day of year: unusable, skip
        valid.push({
          name: item.name,
          label: typeof item.label === "string" ? item.label : item.name,
          day_of_year: Number.isFinite(doyRaw) ? doyRaw : 0,
          doy_sincos: hasSincos ? [Number(item.doy_sincos[0]), Number(item.doy_sincos[1])] : doyToSinCos(doyRaw),
          init_states: item.init_states,
        });
      }
      if (valid.length > 0) return valid;
      lastProblem = "no valid [3][6][32][36] windows in file";
    } catch (e) {
      if ((e as Error).name === "AbortError") throw e;
      lastProblem = (e as Error).message || String(e);
    }
  }
  throw new DemoWindowsError(`data/demo_windows.json missing or invalid (${lastProblem})`);
}
