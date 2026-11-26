import type { VarKey } from "./grid";

export type RGB = [number, number, number];

const hex = (h: string): RGB => [
  parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16),
];

function ramp(stops: string[]) {
  const cols = stops.map(hex);
  return (t: number): RGB => {
    const x = Math.max(0, Math.min(0.999999, t)) * (cols.length - 1);
    const i = Math.floor(x), f = x - i;
    const a = cols[i], b = cols[Math.min(cols.length - 1, i + 1)];
    return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
  };
}

/* Sequential ramps for absolute (Prime / Branched) fields — paper-light → storybook saturation. */
export const SEQUENTIAL: Record<VarKey, (t: number) => RGB> = {
  t2m: ramp(["#bfe3f2", "#8fd0de", "#a8cf6a", "#f2cf5b", "#f0a03c", "#d95f2b", "#a8271c"]),
  tp: ramp(["#f0e2b6", "#cfd98a", "#7cb85c", "#3f9e8f", "#2f7fb5", "#274f9e", "#5b3a8e"]),
  tcwv: ramp(["#f3ead0", "#cfe6d8", "#9fd4c8", "#5fb3b3", "#2f8fae", "#1d5f8e"]),
  u10: ramp(["#f5eed8", "#cfe3d0", "#8fc4b8", "#4a9ab8", "#1d5f8e"]),
  v10: ramp(["#f5eed8", "#ddd0e8", "#b9a7d9", "#8a6fb5", "#5b3a8e"]),
  msl: ramp(["#5b3a8e", "#8a6fb5", "#f3e9cf", "#7fb6d9", "#1d6fa3"]),
};

/* Diverging ramp for the Difference Map: deep blue (−) ↔ paper (0) ↔ brick (+). */
export const DIVERGING_COOL = ramp(["#f3e9cf", "#9fc4dd", "#4a9ab8", "#1d6fa3", "#123f66"]);
export const DIVERGING_WARM = ramp(["#f3e9cf", "#f0c98a", "#f0a03c", "#d95f2b", "#a8271c"]);

export function divergingColor(t: number): RGB {
  if (t < 0) return DIVERGING_COOL(Math.min(1, -t));
  return DIVERGING_WARM(Math.min(1, t));
}

/* Moist/dry diverging ramp used by the flood lens (dry earth ↔ paper ↔ monsoon blue). */
export const DIVERGING_MOIST = ramp(["#8a5a24", "#c07a08", "#f2cf5b", "#f3e9cf", "#8fd0de", "#2f7fb5", "#123f66"]);
export function moistColor(t: number): RGB {
  return DIVERGING_MOIST((Math.max(-1, Math.min(1, t)) + 1) / 2);
}

/* Heat-stress ramp for the heatwave lens. */
export const HEAT_RAMP = ramp(["#8fd0de", "#7cb85c", "#f2cf5b", "#f0a03c", "#d95f2b", "#7a1410"]);
/* Energy/crop ramp: loss (brick) → paper → gain (leaf). */
export const ENERGY_RAMP = ramp(["#a8271c", "#d95f2b", "#f0c98a", "#f3e9cf", "#a8cf6a", "#4c8b3f", "#1f4d24"]);

export const rgbCss = ([r, g, b]: RGB, a = 1) =>
  a >= 1 ? `rgb(${r | 0},${g | 0},${b | 0})` : `rgba(${r | 0},${g | 0},${b | 0},${a})`;

/* Fixed sensible display ranges for absolute fields (in backend physical units; msl in Pa). */
export const ABS_RANGE: Record<VarKey, [number, number]> = {
  t2m: [275, 308],
  tp: [0, 40],
  tcwv: [8, 66],
  u10: [-10, 12],
  v10: [-10, 12],
  msl: [99800, 101200],
};

export function diffScale(values: Float32Array, floors: Record<string, number>, key: string, mask?: number[]): number {
  const abs: number[] = [];
  for (let i = 0; i < values.length; i++) {
    if (mask && !mask[i]) continue;
    const a = Math.abs(values[i]);
    if (a > 1e-6) abs.push(a);
  }
  if (!abs.length) return floors[key] ?? 1;
  abs.sort((a, b) => a - b);
  const p98 = abs[Math.min(abs.length - 1, Math.floor(abs.length * 0.98))];
  return Math.max(floors[key] ?? 1, p98);
}

export const DIFF_FLOOR: Record<VarKey, number> = {
  t2m: 0.25, tp: 0.4, tcwv: 0.3, u10: 0.25, v10: 0.25, msl: 12,
};

export function legendStops(scale: number, unit: string) {
  return [-1, -0.5, 0, 0.5, 1].map((t) => ({
    t,
    value: t * scale,
    color: rgbCss(divergingColor(t)),
    label: `${t === 0 ? "0" : (t * scale).toFixed(Math.abs(scale) < 1 ? 2 : 1)}`,
    unit,
  }));
}
