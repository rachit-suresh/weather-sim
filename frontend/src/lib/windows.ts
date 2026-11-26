import { GRID, NC, NR, VAR_KEYS } from "./grid";

export interface DemoWindow {
  name: string;
  label: string;
  day_of_year: number;
  doy_sincos: [number, number];
  /** [3 historical consecutive days][6 channels][32 lat rows][36 lon cols] */
  init_states: number[][][][];
  isFixture?: boolean;
}

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
 * Loads historical 2020 input windows from `data/demo_windows.json` when hosted
 * alongside the backend static assets. If unavailable locally (standalone dev),
 * returns clearly labelled offline development fixture windows with valid
 * [3][6][32][36] physical units so live /simulate calls remain testable.
 */
export async function loadDemoWindows(signal?: AbortSignal): Promise<{ windows: DemoWindow[]; fromFixture: boolean }> {
  for (const path of ["./data/demo_windows.json", "/data/demo_windows.json"]) {
    try {
      const res = await fetch(path, { signal });
      if (!res.ok) continue;
      const raw = await res.json();
      const arr = Array.isArray(raw) ? raw : Array.isArray(raw?.windows) ? raw.windows : [];
      const valid: DemoWindow[] = [];
      for (const item of arr) {
        if (!item || typeof item.name !== "string" || !isValidState3x6x32x36(item.init_states)) continue;
        const doy = Number.isFinite(Number(item.day_of_year ?? item.doy)) ? Number(item.day_of_year ?? item.doy) : 196;
        const sincos: [number, number] =
          Array.isArray(item.doy_sincos) && item.doy_sincos.length === 2
            ? [Number(item.doy_sincos[0]), Number(item.doy_sincos[1])]
            : doyToSinCos(doy);
        valid.push({
          name: item.name,
          label: typeof item.label === "string" ? item.label : item.name,
          day_of_year: doy,
          doy_sincos: sincos,
          init_states: item.init_states,
          isFixture: false,
        });
      }
      if (valid.length > 0) return { windows: valid, fromFixture: false };
    } catch (e) {
      if ((e as Error).name === "AbortError") throw e;
    }
  }
  return { windows: buildOfflineDevFixtures(), fromFixture: true };
}

/**
 * Clearly labelled offline development fixture windows (5 seasonal 2020-style
 * [3][6][32][36] physical-unit inputs). Used ONLY when data/demo_windows.json
 * is absent. Never presented as real ERA5/historical archive data or forecasts.
 */
function buildOfflineDevFixtures(): DemoWindow[] {
  const defs = [
    { name: "jul_monsoon", label: "Jul Monsoon Onset (Dev Fixture)", doy: 196, tempShift: 0, rainMult: 1.15, jetMult: 1.1 },
    { name: "may_premonsoon", label: "May Pre-Monsoon Heat (Dev Fixture)", doy: 140, tempShift: 4.5, rainMult: 0.25, jetMult: 0.55 },
    { name: "aug_active", label: "Aug Active Break Cycle (Dev Fixture)", doy: 228, tempShift: -0.6, rainMult: 1.35, jetMult: 1.2 },
    { name: "oct_retreat", label: "Oct NE Monsoon Transition (Dev Fixture)", doy: 290, tempShift: -2.2, rainMult: 0.65, jetMult: -0.4 },
    { name: "jan_winter", label: "Jan Western Disturbance (Dev Fixture)", doy: 18, tempShift: -9.5, rainMult: 0.2, jetMult: -0.65 },
  ];

  const clamp = (v: number, a: number, b: number) => (v < a ? a : v > b ? b : v);

  return defs.map((def) => {
    const init_states: number[][][][] = Array.from({ length: 3 }, (_, dayIdx) => {
      const dayDrift = (dayIdx - 2) * 0.35;
      const channels = VAR_KEYS.map(() =>
        Array.from({ length: NR }, () => new Array<number>(NC).fill(0)),
      );
      for (let r = 0; r < NR; r++) {
        const la = GRID.lat[r];
        for (let c = 0; c < NC; c++) {
          const lo = GRID.lon[c];
          const ocean = GRID.mask[r][c] === 0;
          const e = GRID.elevation[r][c];

          const heatNW = Math.exp(-0.5 * ((la - 27.5) / 5.5) ** 2) * Math.exp(-0.5 * ((lo - 73) / 6) ** 2);
          const drySE = Math.exp(-0.5 * ((la - 12) / 3) ** 2) * Math.exp(-0.5 * ((lo - 79) / 1.6) ** 2);
          const cloudW = Math.exp(-0.5 * ((lo - 73.8) / 1.4) ** 2) * Math.exp(-0.5 * ((la - 14) / 7) ** 2);
          const cloudNE = Math.exp(-0.5 * ((la - 26) / 2.2) ** 2) * Math.exp(-0.5 * ((lo - 92) / 3) ** 2);

          const t2m = (ocean
            ? 301.2 - 0.12 * (la - 8) + def.tempShift * 0.35
            : 300.8 + def.tempShift + 6.5 * heatNW + 1.8 * drySE - 1.5 * cloudW - 1.3 * cloudNE - 19.0 * e) + dayDrift * 0.25;

          const ghats = Math.exp(-0.5 * ((lo - 73.6) / 1.15) ** 2) * Math.exp(-0.5 * ((la - 15) / 6.5) ** 2);
          const bay = Math.exp(-0.5 * ((la - 16) / 5.5) ** 2) * Math.exp(-0.5 * ((lo - 89) / 4.5) ** 2);
          const ne = Math.exp(-0.5 * ((la - 25.1) / 1.0) ** 2) * Math.exp(-0.5 * ((lo - 91.8) / 1.9) ** 2);
          const trough = Math.exp(-0.5 * ((la - 26) / 3.0) ** 2) * (lo > 76 && lo < 89 ? 1 : 0.25);
          let tp = (1.6 + 28 * ghats + 14 * bay + 32 * ne + 8 * trough) * def.rainMult;
          if (ocean) tp *= 0.85;
          if (lo < 74 && la > 23) tp = Math.max(0.3, (1.2 + 0.5 * Math.max(0, 30 - la)) * def.rainMult);
          if (lo > 77 && lo < 80.5 && la < 16) tp *= 0.35;
          if (e > 0.7) tp *= 0.5;
          tp = Math.max(0, tp + dayDrift * 0.4);

          const jet = Math.exp(-0.5 * ((la - 15) / 9) ** 2);
          let u10 = (7.0 * jet + 0.9) * def.jetMult;
          let v10 = (4.4 * jet + 0.5) * def.jetMult;
          const tr = Math.exp(-0.5 * ((la - 26.5) / 2.4) ** 2);
          u10 -= 6.5 * tr * Math.max(0.2, def.jetMult);
          v10 -= 1.0 * tr;
          const drag = (ocean ? 1.0 : 0.62) * (1 - 0.55 * e);
          u10 = u10 * drag + dayDrift * 0.2;
          v10 = v10 * drag + dayDrift * 0.15;

          // msl in Pa!
          let mslHpa = 1006.8 - 4.2 * Math.exp(-0.5 * ((la - 26.5) / 3.2) ** 2) + (def.tempShift < 0 ? 5.5 : 0);
          if (ocean) mslHpa += 1.2;
          mslHpa -= 0.5 * tp / 10;
          const msl = (mslHpa + dayDrift * 0.3) * 100;

          let tcwv = (ocean ? 57 - 0.35 * (la - 8) : 55 - 0.52 * (la - 8) - 25 * e) * (0.55 + 0.45 * clamp(def.rainMult, 0.3, 1.3));
          if (lo < 74 && la > 23) tcwv -= 11;
          tcwv = clamp(tcwv + dayDrift * 0.5, 5, 68);

          channels[0][r][c] = +t2m.toFixed(3);
          channels[1][r][c] = +tp.toFixed(3);
          channels[2][r][c] = +u10.toFixed(3);
          channels[3][r][c] = +v10.toFixed(3);
          channels[4][r][c] = +msl.toFixed(1);
          channels[5][r][c] = +tcwv.toFixed(3);
        }
      }
      return channels;
    });

    return {
      name: def.name,
      label: def.label,
      day_of_year: def.doy,
      doy_sincos: doyToSinCos(def.doy),
      init_states,
      isFixture: true,
    };
  });
}
