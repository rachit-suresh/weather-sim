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

/* ── spec IMP-5: the user's own input file ─────────────────────────────────── */

/** Thrown with a plain, user-facing message naming the first problem in an input file. */
export class InputFileError extends Error {}

const CHANNELS = "t2m, tp, u10, v10, msl, tcwv";

/** Returns the first shape/number problem in init_states, or null when it is a valid [3][6][32][36]. */
function firstStateProblem(states: unknown): string | null {
  if (!Array.isArray(states)) return "init_states is missing or not a list";
  if (states.length !== 3) return `init_states has ${states.length} days, expected 3 (oldest first)`;
  for (let d = 0; d < 3; d++) {
    const s = states[d];
    if (!Array.isArray(s) || s.length !== 6) {
      return `init_states[${d}] has ${Array.isArray(s) ? s.length : 0} channels, expected 6 (${CHANNELS})`;
    }
    for (let ch = 0; ch < 6; ch++) {
      const rows = s[ch];
      if (!Array.isArray(rows) || rows.length !== NR) {
        return `init_states[${d}][${ch}] has ${Array.isArray(rows) ? rows.length : 0} rows, expected ${NR}`;
      }
      for (let r = 0; r < NR; r++) {
        const cols = rows[r];
        if (!Array.isArray(cols) || cols.length !== NC) {
          return `init_states[${d}][${ch}][${r}] has ${Array.isArray(cols) ? cols.length : 0} columns, expected ${NC}`;
        }
        for (let c = 0; c < NC; c++) {
          if (typeof cols[c] !== "number" || !Number.isFinite(cols[c])) {
            return `init_states[${d}][${ch}][${r}][${c}] is not a finite number`;
          }
        }
      }
    }
  }
  return null;
}

function dateToDoy(text: string): number | null {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(text.trim());
  if (!m) return null;
  const t = Date.UTC(Number(m[1]), Number(m[2]) - 1, Number(m[3]));
  const d = new Date(t);
  if (d.getUTCMonth() !== Number(m[2]) - 1 || d.getUTCDate() !== Number(m[3])) return null;
  return Math.round((t - Date.UTC(Number(m[1]), 0, 1)) / 86400000) + 1;
}

/** One window object -> DemoWindow, or an InputFileError naming the problem (prefix = "window 2: " etc). */
function parseOneWindow(item: unknown, fallbackName: string, prefix: string): DemoWindow {
  if (!item || typeof item !== "object" || Array.isArray(item)) {
    throw new InputFileError(`${prefix}expected an object with init_states and a date`);
  }
  const o = item as Record<string, unknown>;
  const problem = firstStateProblem(o.init_states);
  if (problem) throw new InputFileError(prefix + problem);

  let doy: number | null = null;
  let sincos: [number, number] | null = null;
  if (Array.isArray(o.doy_sincos)) {
    const [s, c] = o.doy_sincos.map(Number);
    if (o.doy_sincos.length !== 2 || !Number.isFinite(s) || !Number.isFinite(c) || Math.hypot(s, c) < 0.5) {
      throw new InputFileError(`${prefix}doy_sincos must be [sin, cos] of the last input day`);
    }
    sincos = [s, c];
    const ang = Math.atan2(s, c);
    doy = Math.round(((ang < 0 ? ang + 2 * Math.PI : ang) * 365) / (2 * Math.PI)) || 365;
  } else if (o.day_of_year !== undefined) {
    const v = Number(o.day_of_year);
    if (!Number.isFinite(v) || v < 1 || v > 366) throw new InputFileError(`${prefix}day_of_year must be 1-366`);
    doy = Math.round(v);
  } else if (typeof o.date === "string") {
    doy = dateToDoy(o.date);
    if (doy === null) throw new InputFileError(`${prefix}date must look like 2020-05-17 (the last input day)`);
  } else {
    throw new InputFileError(`${prefix}no date: add "date": "YYYY-MM-DD", "day_of_year" or "doy_sincos" for the last input day`);
  }

  const name = typeof o.name === "string" && o.name.trim() ? o.name.trim().slice(0, 40) : fallbackName;
  return {
    name,
    label: "custom",
    day_of_year: doy,
    doy_sincos: sincos ?? doyToSinCos(doy),
    init_states: o.init_states as number[][][][],
  };
}

/**
 * spec IMP-5 (a)+(b): parses the text of a user's .json file. Accepts one window object (the fields of a /simulate
 * request), a list of them, or {"windows": [...]}. Throws InputFileError with the first problem found.
 */
export function parseInputFile(text: string, fileName: string): DemoWindow[] {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    throw new InputFileError("the file is not valid JSON");
  }
  const base = fileName.replace(/\.json$/i, "") || "input";
  const list = Array.isArray(raw)
    ? raw
    : raw && typeof raw === "object" && Array.isArray((raw as { windows?: unknown }).windows)
      ? (raw as { windows: unknown[] }).windows
      : [raw];
  if (list.length === 0) throw new InputFileError("the file holds no windows");
  return list.map((item, i) =>
    parseOneWindow(item, list.length > 1 ? `${base} ${i + 1}` : base, list.length > 1 ? `window ${i + 1}: ` : ""));
}

/** spec IMP-5 (d): the current window as a loadable .json (same fields as a /simulate request). */
export function windowToJson(w: DemoWindow): string {
  return JSON.stringify({
    name: w.name.replace(/^Your input: /, ""),
    day_of_year: w.day_of_year,
    doy_sincos: w.doy_sincos,
    units: { t2m: "K", tp: "mm/day", u10: "m/s", v10: "m/s", msl: "Pa", tcwv: "kg/m2" },
    channels: ["t2m", "tp", "u10", "v10", "msl", "tcwv"],
    init_states: w.init_states,
  });
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
