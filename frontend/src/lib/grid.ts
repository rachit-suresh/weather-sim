import gridJson from "../../shared/grid.json";
import indiaBordersJson from "../assets/india-borders.json";

/** Fixed backend channel order — never reorder. */
export type VarKey = "t2m" | "tp" | "u10" | "v10" | "msl" | "tcwv";
export const VAR_KEYS: VarKey[] = ["t2m", "tp", "u10", "v10", "msl", "tcwv"];
export const VAR_INDEX: Record<VarKey, number> = {
  t2m: 0, tp: 1, u10: 2, v10: 3, msl: 4, tcwv: 5,
};

export interface Region { name: string; bbox: number[]; cells: [number, number][] }
export interface City { name: string; lat: number; lon: number; note: string; r: number; c: number; i: number }

export interface GridSpec {
  nlat: number; nlon: number; step: number; lat0: number; lon0: number;
  order: string;
  lat: number[]; lon: number[];
  mask: number[][];
  land_sea_mask: number[][];
  elevation: number[][];
  regions: Region[];
  cities: City[];
  channels: VarKey[];
  units: Record<VarKey, string>;
  labels: Record<VarKey, string>;
  max_steps: number;
}

export interface BorderFeature {
  type: "Feature";
  properties: { name: string; kind: "national" | "island" | "neighbor_coast" };
  geometry:
    | { type: "Polygon"; coordinates: [number, number][][] }
    | { type: "MultiLineString"; coordinates: [number, number][][] };
}
export const INDIA_BORDERS = indiaBordersJson as unknown as {
  type: "FeatureCollection";
  features: BorderFeature[];
};

export const GRID = gridJson as unknown as GridSpec;
export const NR = GRID.nlat; // 32 rows: 0 = 6.5°N (south) .. 31 = 37.5°N (north)
export const NC = GRID.nlon; // 36 cols: 0 = 65.5°E (west) .. 35 = 100.5°E (east)
export const CELLS = NR * NC;

export const idx = (r: number, c: number) => r * NC + c;
export const rowOf = (i: number) => Math.floor(i / NC);
export const colOf = (i: number) => i % NC;
export const inBounds = (r: number, c: number) => r >= 0 && r < NR && c >= 0 && c < NC;

/** Local reference land/India classification (aligned to 6.5..37.5°N, 65.5..100.5°E). */
export const isOcean = (r: number, c: number) => GRID.mask[r][c] === 0;
export const isIndia = (r: number, c: number) => GRID.mask[r][c] === 2;
export const isLand = (r: number, c: number) => GRID.mask[r][c] !== 0;

/** All India cell indices — used for regional impact aggregation. */
export const INDIA_CELLS: number[] = (() => {
  const out: number[] = [];
  for (let r = 0; r < NR; r++) for (let c = 0; c < NC; c++) if (GRID.mask[r][c] === 2) out.push(idx(r, c));
  return out;
})();

export const cellLatLon = (r: number, c: number) => ({ lat: GRID.lat[r], lon: GRID.lon[c] });
export const latLonToCell = (lat: number, lon: number): [number, number] => [
  Math.max(0, Math.min(NR - 1, Math.round((lat - GRID.lat0) / GRID.step))),
  Math.max(0, Math.min(NC - 1, Math.round((lon - GRID.lon0) / GRID.step))),
];

export const fmtLat = (lat: number) => `${Math.abs(lat).toFixed(1)}°${lat >= 0 ? "N" : "S"}`;
export const fmtLon = (lon: number) => `${Math.abs(lon).toFixed(1)}°${lon >= 0 ? "E" : "W"}`;

export function regionOf(r: number, c: number, landSeaValue?: number): string {
  const { lat, lon } = cellLatLon(r, c);
  for (const reg of GRID.regions) {
    const [a, b, x, y] = reg.bbox;
    if (lat >= a && lat <= b && lon >= x && lon <= y) return reg.name;
  }
  const sea = landSeaValue !== undefined ? landSeaValue < 0.25 : isOcean(r, c);
  if (sea) return "Open water";
  if (isIndia(r, c)) {
    let best = GRID.regions[0].name;
    let bd = Infinity;
    for (const reg of GRID.regions) {
      const [a, b, x, y] = reg.bbox;
      const d = Math.hypot(lat - (a + b) / 2, lon - (x + y) / 2);
      if (d < bd) { bd = d; best = reg.name; }
    }
    return best;
  }
  return "Regional land";
}

export function nearestCity(r: number, c: number): { city: City; distanceCells: number } | null {
  let best: City | null = null;
  let bd = Infinity;
  for (const city of GRID.cities) {
    const d = Math.hypot(city.r - r, city.c - c);
    if (d < bd) { bd = d; best = city; }
  }
  return best ? { city: best, distanceCells: bd } : null;
}

export const VAR_META: Record<
  VarKey,
  { short: string; label: string; backendUnit: string; displayUnit: string; icon: string }
> = {
  t2m:  { short: "T2M",  label: "2 m temperature",      backendUnit: "K",      displayUnit: "°C",     icon: "thermometer" },
  tp:   { short: "TP",   label: "Total precipitation",  backendUnit: "mm/day", displayUnit: "mm/day", icon: "cloud-rain" },
  u10:  { short: "U10",  label: "10 m eastward wind",   backendUnit: "m/s",    displayUnit: "m/s",    icon: "wind" },
  v10:  { short: "V10",  label: "10 m northward wind",  backendUnit: "m/s",    displayUnit: "m/s",    icon: "wind" },
  msl:  { short: "MSL",  label: "Sea-level pressure",   backendUnit: "Pa",     displayUnit: "hPa",    icon: "gauge" },
  tcwv: { short: "TCWV", label: "Column water vapour",  backendUnit: "kg/m²",  displayUnit: "kg/m²",  icon: "droplets" },
};

/**
 * Convert raw backend value (in physical units) to user-facing display units.
 * Never apply this before sending requests or computing backend deltas!
 */
export function display(v: VarKey, raw: number): number {
  if (v === "t2m") return raw - 273.15;
  if (v === "msl") return raw / 100;
  return raw;
}

export function displayUnit(v: VarKey): string {
  return VAR_META[v].displayUnit;
}

/** Convert a raw backend delta to display-unit delta (K diff == °C diff; Pa diff / 100 == hPa diff). */
export function displayDelta(v: VarKey, rawDelta: number): number {
  return v === "msl" ? rawDelta / 100 : rawDelta;
}

/** Convert a user-entered UI delta (in displayUnit) into raw backend physical-unit delta. */
export function uiDeltaToBackend(v: VarKey, uiDelta: number): number {
  return v === "msl" ? uiDelta * 100 : uiDelta;
}

/**
 * Wind speed & direction from eastward (u10) and northward (v10) components.
 * Documents both conventions:
 * - towardDeg: azimuth clockwise from North toward which the wind blows (atan2(u, v))
 * - fromDeg: meteorological origin direction FROM which wind blows (atan2(-u, -v))
 */
export function windMetrics(u10: number, v10: number) {
  const speed = Math.hypot(u10, v10);
  const towardDeg = ((Math.atan2(u10, v10) * 180) / Math.PI + 360) % 360;
  const fromDeg = (towardDeg + 180) % 360;
  const dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];
  const towardCompass = dirs[Math.round(towardDeg / 45) % 8];
  const fromCompass = dirs[Math.round(fromDeg / 45) % 8];
  return { speed, towardDeg, fromDeg, towardCompass, fromCompass };
}
