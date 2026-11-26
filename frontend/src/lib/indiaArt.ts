/* ════════════════════════════════════════════════════════════════════════════
   India art map (spec IMP-3a) — a storybook pixel-art terrain raster of the
   forecast grid, generated once in the browser (no assets, works offline).
   Coast, islands and India's outline come from lat/lon outlines in indiaGeo.ts (spec IMP-3h).

   Extent = the 32 x 36 forecast grid: lat 6..38 N, lon 65..101 E, at ART_P art
   pixels per degree, north up. Elevation (from the grid, read through noise-
   warped coordinates so no 1-deg edges show) drives the biomes and mountains;
   ranges and rivers are drawn from hand-placed lat/lon polylines.
   ════════════════════════════════════════════════════════════════════════════ */
import { GRID, NC, NR } from "@/lib/grid";
import { geoKind } from "@/lib/indiaGeo";

export const ART_P = 10;                       // art pixels per degree
export const ART_W = NC * ART_P;               // 360
export const ART_H = NR * ART_P;               // 320
export const ART_LAT0 = 6, ART_LAT1 = 38, ART_LON0 = 65, ART_LON1 = 101;

type RGB = [number, number, number];

/* ── palette (taken from the pixel-art country-map reference) ── */
const C = {
  seaDeep: [36, 137, 180] as RGB,
  sea: [43, 157, 199] as RGB,
  seaShallow: [74, 182, 216] as RGB,
  wave: [233, 247, 251] as RGB,
  surf: [246, 252, 255] as RGB,
  cliffTop: [101, 66, 34] as RGB,
  cliffBot: [62, 40, 19] as RGB,
  plain: [226, 198, 90] as RGB,
  plainAlt: [216, 186, 79] as RGB,
  ganga: [206, 205, 96] as RGB,
  deccan: [214, 180, 84] as RGB,
  forest: [114, 166, 66] as RGB,
  highland: [184, 166, 96] as RGB,
  rock: [160, 157, 147] as RGB,
  plateau: [200, 190, 162] as RGB,               // high flat stony upland (Tibet, Pamir)
  sand: [236, 214, 152] as RGB,
  dune: [212, 184, 116] as RGB,
  river: [58, 142, 208] as RGB,
  riverLight: [116, 190, 232] as RGB,
  foreign: [214, 200, 162] as RGB,
  border: [120, 72, 40] as RGB,
};

/* ── sprites (art-pixel units) ── */
const INK: Record<string, RGB> = {
  o: [58, 53, 48], W: [255, 255, 255], l: [214, 214, 206], m: [163, 163, 155], d: [116, 116, 108],
  G: [148, 200, 88], g: [94, 154, 58], k: [60, 118, 42], t: [107, 68, 35], K: [42, 63, 28],
  h: [176, 152, 92], H: [140, 118, 66], s: [212, 184, 116], c: [77, 140, 60],
  r: [196, 58, 42], R: [120, 32, 24], w: [250, 250, 245], b: [72, 52, 40],
};
const SPR = {
  peak: ["....o....", "...oWo...", "..oWWlo..", "..oWllmo.", ".olllmmo.", ".ollmmmdo", "olmmmmddo", "ooooooooo"],
  peakSmall: ["...o...", "..oWo..", ".oWlmo.", ".olmmo.", "olmmddo", "ooooooo"],
  hill: ["..oo...", ".ohhHo.", "ohhhHHo", "ooooooo"],
  tree: [".KKK.", "KGgGK", "KggkK", "KgkkK", ".KkK.", "..t.."],
  treeSmall: [".KK.", "KGgK", "KgkK", ".tt."],
  pine: ["..K..", ".KgK.", ".KGK.", "KgggK", "KGgkK", "..t.."],
  dune: [".sss..", "s...ss"],
  cactus: [".c.", "ccc", ".c.", ".c."],
  tuft: ["c.c", ".c."],
  boat: ["...w...", "...ww..", "...www.", "...w...", "RrrrrrR", ".RRRRR."],
  boatDark: ["...w..", "...ww.", "...w..", "bbbbbb", ".bbbb."],
} as const;

/* ── hand-placed geography (lat, lon) ── */
const RIVERS: { w: number; pts: [number, number][] }[] = [
  { w: 2, pts: [[30.9, 78.9], [29.95, 78.16], [28.4, 79.0], [27.05, 79.9], [26.45, 80.33], [25.43, 81.85], [25.32, 83.0], [25.6, 85.1], [25.25, 86.98], [24.81, 87.93], [23.4, 88.3], [22.57, 88.36], [21.7, 88.2]] }, // Ganga
  { w: 1, pts: [[31.0, 78.45], [30.3, 77.6], [28.61, 77.25], [27.18, 78.02], [26.78, 79.02], [25.42, 81.88]] }, // Yamuna
  { w: 2, pts: [[28.5, 95.9], [27.48, 94.9], [26.63, 92.8], [26.18, 91.75], [26.02, 89.98], [25.2, 89.7], [23.8, 89.75], [22.6, 90.5]] }, // Brahmaputra
  { w: 2, pts: [[34.16, 77.58], [34.9, 76.2], [35.3, 75.0], [34.5, 72.9], [33.9, 72.2], [31.5, 71.2], [29.0, 70.0], [27.7, 68.86], [25.4, 68.37], [24.0, 67.5]] }, // Indus
  { w: 1, pts: [[22.67, 81.75], [23.17, 79.95], [22.75, 77.73], [22.24, 76.15], [21.9, 74.5], [21.7, 72.97], [21.6, 72.6]] }, // Narmada
  { w: 1, pts: [[20.0, 73.8], [19.6, 75.3], [19.15, 77.3], [18.9, 79.5], [17.67, 80.89], [17.0, 81.8], [16.6, 82.3]] }, // Godavari
  { w: 1, pts: [[17.92, 73.66], [16.85, 74.57], [16.3, 76.3], [16.08, 78.87], [16.51, 80.62], [15.9, 81.0]] }, // Krishna
  { w: 1, pts: [[12.38, 75.49], [12.42, 76.6], [12.12, 77.78], [10.83, 78.7], [10.9, 79.3], [11.1, 79.85]] }, // Kaveri
  { w: 1, pts: [[22.9, 82.0], [21.8, 83.4], [21.0, 84.5], [20.5, 85.6], [20.3, 86.6]] }, // Mahanadi
];
const RANGES: { kind: "hill" | "peakSmall"; pts: [number, number][] }[] = [
  { kind: "peakSmall", pts: [[21.0, 73.8], [19.0, 73.6], [17.0, 73.8], [15.0, 74.2], [13.0, 75.3], [11.0, 76.6], [9.0, 77.2]] }, // Western Ghats
  { kind: "hill", pts: [[21.5, 85.3], [19.5, 84.0], [17.6, 82.4], [16.0, 80.0], [14.0, 79.2], [12.0, 78.6]] },                 // Eastern Ghats
  { kind: "hill", pts: [[24.5, 72.7], [25.8, 74.2], [27.2, 75.9], [28.4, 77.0]] },                                            // Aravalli
  { kind: "hill", pts: [[23.4, 75.0], [24.0, 77.5], [24.3, 80.0], [24.6, 82.0]] },                                            // Vindhya
  { kind: "hill", pts: [[21.6, 74.8], [21.9, 77.0], [22.3, 79.5], [22.6, 81.2]] },                                            // Satpura
];
const BOATS: { lat: number; lon: number; dark?: boolean }[] = [
  { lat: 18.6, lon: 68.6 }, { lat: 14.2, lon: 70.6, dark: true }, { lat: 9.6, lon: 73.2 },
  { lat: 16.0, lon: 87.4 }, { lat: 12.4, lon: 84.6, dark: true }, { lat: 19.4, lon: 89.6 }, { lat: 8.0, lon: 80.8 },
];

/* ── helpers ── */
const hash = (x: number, y: number) => (((x * 73856093) ^ (y * 19349663) ^ 0x5bd1e995) >>> 0);
const rand = (x: number, y: number) => (hash(x, y) % 10007) / 10007;
function valueNoise(x: number, y: number, s: number) {
  const gx = Math.floor(x / s), gy = Math.floor(y / s), fx = x / s - gx, fy = y / s - gy;
  const a = rand(gx, gy), b = rand(gx + 1, gy), c = rand(gx, gy + 1), d = rand(gx + 1, gy + 1);
  const u = fx * fx * (3 - 2 * fx), v = fy * fy * (3 - 2 * fy);
  return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
}
/** Bilinear sample of a cell-centred grid field at (lat, lon). */
function bilinear(field: (r: number, c: number) => number, lat: number, lon: number) {
  const fr = Math.max(0, Math.min(NR - 1.0001, lat - 6.5)), fc = Math.max(0, Math.min(NC - 1.0001, lon - 65.5));
  const r0 = Math.floor(fr), c0 = Math.floor(fc), dr = fr - r0, dc = fc - c0;
  return field(r0, c0) * (1 - dr) * (1 - dc) + field(r0, c0 + 1) * (1 - dr) * dc
    + field(r0 + 1, c0) * dr * (1 - dc) + field(r0 + 1, c0 + 1) * dr * dc;
}
const toX = (lon: number) => (lon - ART_LON0) * ART_P;
const toY = (lat: number) => (ART_LAT1 - lat) * ART_P;
function distToPolyline(lat: number, lon: number, pts: [number, number][]) {
  let best = 1e9;
  for (let k = 0; k + 1 < pts.length; k++) {
    const [a0, b0] = pts[k], [a1, b1] = pts[k + 1];
    const dx = a1 - a0, dy = b1 - b0, L = dx * dx + dy * dy || 1;
    const t = Math.max(0, Math.min(1, ((lat - a0) * dx + (lon - b0) * dy) / L));
    best = Math.min(best, Math.hypot(lat - (a0 + t * dx), lon - (b0 + t * dy)));
  }
  return best;
}

export type ArtKind = 0 | 1 | 2;                // 0 sea, 1 neighbour land, 2 India
export interface IndiaArt {
  canvas: HTMLCanvasElement;
  rgba: Uint8ClampedArray;                      // ART_W * ART_H * 4
  kind: Uint8Array;                             // ART_W * ART_H
}

let cached: IndiaArt | null = null;

/** Art-map pixel index for a geographic point, or -1 outside the grid. */
export function artIndex(lat: number, lon: number): number {
  if (lat < ART_LAT0 || lat >= ART_LAT1 || lon < ART_LON0 || lon >= ART_LON1) return -1;
  const x = Math.min(ART_W - 1, Math.floor((lon - ART_LON0) * ART_P));
  const y = Math.min(ART_H - 1, Math.floor((ART_LAT1 - lat) * ART_P));
  return y * ART_W + x;
}

export function getIndiaArt(): IndiaArt {
  if (cached) return cached;
  const W = ART_W, H = ART_H, N = W * H;
  const rgba = new Uint8ClampedArray(N * 4);
  const kind = new Uint8Array(N);
  const biome = new Uint8Array(N);              // 0 sea,1 plain,2 ganga,3 deccan,4 forest,5 highland,6 mountain,7 sand,8 plateau
  const elevAt = new Float32Array(N);
  const put = (i: number, c: RGB, a = 1) => {
    const o = i * 4;
    rgba[o] = rgba[o] * (1 - a) + c[0] * a; rgba[o + 1] = rgba[o + 1] * (1 - a) + c[1] * a;
    rgba[o + 2] = rgba[o + 2] * (1 - a) + c[2] * a; rgba[o + 3] = 255;
  };

  // 1) land / India / biome per art pixel
  for (let y = 0; y < H; y++) {
    const lat = ART_LAT1 - (y + 0.5) / ART_P;
    for (let x = 0; x < W; x++) {
      const lon = ART_LON0 + (x + 0.5) / ART_P;
      const i = y * W + x;
      const n2 = valueNoise(x + 999, y + 333, 3) - 0.5;
      // spec IMP-3h: land/sea and India/neighbour from hand-entered lat/lon outlines, not the 1-deg mask
      const k = geoKind(lat, lon);
      if (k === 0) { kind[i] = 0; continue; }
      kind[i] = k;
      // biome rules read noise-warped coordinates so their edges wander instead of following 1-deg lines
      // (large swirl + medium wobble + per-pixel dither, so zone edges look hand-drawn, not ruled)
      const la = lat + 2.0 * (valueNoise(x + 500, y + 100, 30) - 0.5) + 0.7 * (valueNoise(x + 71, y + 13, 9) - 0.5) + 0.35 * (rand(x, y) - 0.5);
      const lo = lon + 2.0 * (valueNoise(x + 200, y + 800, 30) - 0.5) + 0.7 * (valueNoise(x + 19, y + 57, 9) - 0.5) + 0.35 * (rand(y + 7, x) - 0.5);
      const elev = (a: number, b: number) => bilinear((r, c) => GRID.elevation[r][c], a, b);
      const e = Math.max(0, elev(la, lo) + 0.06 * n2);
      // steepness (per degree): mountain sprites belong on steep walls (Himalaya), not on flat high plateaus (Tibet)
      const slope = Math.hypot(elev(la + 0.6, lo) - elev(la - 0.6, lo), elev(la, lo + 0.6) - elev(la, lo - 0.6)) / 1.2;
      elevAt[i] = e;
      let b = 1;
      if (e >= 0.5) b = slope > 0.07 || (kind[i] === 2 && e < 0.8) ? 6 : 8;
      else if (la >= 24 && la <= 30.6 && lo >= 69 && lo <= 75.6 && e < 0.22) b = 7;
      else if (e >= 0.3) b = 5;
      else if ((la > 21.5 && lo > 89.3) || (la < 21 && distToPolyline(la, lo, RANGES[0].pts) < 1.0)
        || (la >= 18 && la <= 24.5 && lo >= 79 && lo <= 86.5 && e >= 0.1) || (la > 27 && e >= 0.16)) b = 4;
      else if (la >= 24 && la <= 30.8 && lo >= 75.6 && lo <= 88.8 && e < 0.14) b = 2;
      else if (la >= 13 && la < 21 && lo >= 74.5 && lo <= 80.5 && e >= 0.1) b = 3;
      biome[i] = b;
    }
  }

  // 2) base colours
  const shallowNear = (x: number, y: number) => {
    for (let dy = -3; dy <= 3; dy++) for (let dx = -3; dx <= 3; dx++) {
      const xx = x + dx, yy = y + dy;
      if (xx >= 0 && yy >= 0 && xx < W && yy < H && kind[yy * W + xx] !== 0) return true;
    }
    return false;
  };
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = y * W + x;
    const grain = (rand(x, y) - 0.5) * 0.06;
    if (kind[i] === 0) {
      const deep = valueNoise(x, y, 18);
      const base = shallowNear(x, y) ? C.seaShallow : deep > 0.62 ? C.seaDeep : C.sea;
      put(i, [base[0] * (1 + grain), base[1] * (1 + grain), base[2] * (1 + grain)]);
      continue;
    }
    const b = biome[i];
    let col: RGB = b === 2 ? C.ganga : b === 3 ? C.deccan : b === 4 ? C.forest : b === 5 ? C.highland
      : b === 6 ? C.rock : b === 7 ? C.sand : b === 8 ? C.plateau : (valueNoise(x, y, 7) > 0.55 ? C.plainAlt : C.plain);
    if (kind[i] === 1) col = [col[0] * 0.45 + C.foreign[0] * 0.55, col[1] * 0.45 + C.foreign[1] * 0.55, col[2] * 0.45 + C.foreign[2] * 0.55];
    const k = 1 + grain;
    put(i, [col[0] * k, col[1] * k, col[2] * k]);
  }

  // 3) rivers (land only)
  const plot = (x: number, y: number, c: RGB) => {
    if (x < 0 || y < 0 || x >= W || y >= H) return;
    const i = y * W + x; if (kind[i] !== 0) put(i, c);
  };
  for (const rv of RIVERS) {
    for (let k = 0; k + 1 < rv.pts.length; k++) {
      const [la0, lo0] = rv.pts[k], [la1, lo1] = rv.pts[k + 1];
      const x0 = toX(lo0), y0 = toY(la0), x1 = toX(lo1), y1 = toY(la1);
      const steps = Math.ceil(Math.hypot(x1 - x0, y1 - y0) * 2);
      for (let s = 0; s <= steps; s++) {
        const t = s / steps;
        const wob = (valueNoise(x0 + t * (x1 - x0), y0 + t * (y1 - y0), 6) - 0.5) * 2.4;
        const xx = Math.round(x0 + t * (x1 - x0) + wob), yy = Math.round(y0 + t * (y1 - y0));
        plot(xx, yy, C.river);
        if (rv.w > 1) { plot(xx + 1, yy, C.river); plot(xx, yy + 1, C.riverLight); }
      }
    }
  }

  // 4) India border: dotted line where India meets neighbour land
  for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) {
    const i = y * W + x;
    if (kind[i] !== 2) continue;
    if ((kind[i + 1] === 1 || kind[i - 1] === 1 || kind[i + W] === 1 || kind[i - W] === 1) && ((x + y) & 1) === 0) put(i, C.border, 0.75);
  }

  // 5) coast: brown extruded cliff on south/east faces, white surf elsewhere
  const kindCopy = kind.slice();
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = y * W + x;
    if (kindCopy[i] !== 0) continue;
    const landN1 = y > 0 && kindCopy[i - W] !== 0, landN2 = y > 1 && kindCopy[i - 2 * W] !== 0;
    const landW1 = x > 0 && kindCopy[i - 1] !== 0;
    if (landN1) { put(i, C.cliffTop); continue; }
    if (landN2 || (landW1 && y > 0 && kindCopy[i - W - 1] !== 0)) { put(i, C.cliffBot); continue; }
    const adj = (y < H - 1 && kindCopy[i + W] !== 0) || (x < W - 1 && kindCopy[i + 1] !== 0) || landW1;
    if (adj) put(i, C.surf);
  }

  // 6) sea texture: white wave marks
  for (let y = 4; y < H - 2; y += 7) for (let x = 2 + ((y / 7) % 2) * 5; x < W - 6; x += 11) {
    const jx = x + (hash(x, y) % 5), jy = y + (hash(y, x) % 3);
    const i0 = jy * W + jx;
    if (kind[i0] !== 0 || shallowNear(jx, jy) || hash(jx, jy) % 3 === 0) continue;
    put(i0, C.wave); put(i0 + 1, C.wave, 0.7); put(i0 + 3, C.wave); put(i0 - W + 2, C.wave, 0.8);
  }

  // 7) sprites, sorted north-to-south so southern ones overlap (stacked look)
  const sprites: { name: keyof typeof SPR; x: number; y: number; dim: boolean }[] = [];
  const add = (name: keyof typeof SPR, x: number, y: number) => {
    if (x < 1 || y < 1 || x >= W - 1 || y >= H - 1) return;
    const i = y * W + x; if (kind[i] === 0) return;
    // no sprites on tiny islands (Lakshadweep, Nicobar): they would just read as clutter
    if (kind[Math.max(0, i - 4)] === 0 || kind[Math.min(W * H - 1, i + 4)] === 0 || kind[Math.max(0, i - 4 * W)] === 0 || kind[Math.min(W * H - 1, i + 4 * W)] === 0) return;
    sprites.push({ name, x, y, dim: kind[i] === 1 });
  };
  for (let y = 2; y < H; y += 3) for (let x = 2; x < W; x += 3) {
    const jx = x + (hash(x, y) % 3) - 1, jy = y + (hash(y + 7, x) % 3) - 1;
    if (jx < 0 || jy < 0 || jx >= W || jy >= H) continue;
    const i = jy * W + jx; if (kind[i] === 0) continue;
    const r = rand(jx, jy), b = biome[i], foreignThin = kind[i] === 1 ? 0.45 : 1;
    if (b === 6 && r < 0.55 * foreignThin) add(elevAt[i] > 0.75 || r < 0.3 ? "peak" : "peakSmall", jx, jy);
    else if (b === 8 && r < 0.05) add("peakSmall", jx, jy);
    else if (b === 4 && r < 0.42 * foreignThin) add(r < 0.3 ? "tree" : "treeSmall", jx, jy);
    else if (b === 5 && r < 0.16 * foreignThin) add(jy < 120 ? "pine" : "hill", jx, jy);
    else if (b === 7 && r < 0.06) add(r < 0.012 ? "cactus" : "dune", jx, jy);
    else if ((b === 1 || b === 2 || b === 3) && r < 0.035 * foreignThin) add(r < 0.02 ? "treeSmall" : "tuft", jx, jy);
  }
  for (const rg of RANGES) {
    for (let k = 0; k + 1 < rg.pts.length; k++) {
      const [la0, lo0] = rg.pts[k], [la1, lo1] = rg.pts[k + 1];
      const steps = Math.max(1, Math.round(Math.hypot(la1 - la0, lo1 - lo0) * 1.8));
      for (let s = 0; s < steps; s++) {
        const t = (s + rand(k, s) * 0.5) / steps;
        add(rg.kind, Math.round(toX(lo0 + t * (lo1 - lo0)) + (rand(s, k) - 0.5) * 6), Math.round(toY(la0 + t * (la1 - la0)) + (rand(k + 3, s) - 0.5) * 4));
      }
    }
  }
  sprites.sort((a, b) => a.y - b.y);
  for (const s of sprites) {
    const art = SPR[s.name];
    const left = s.x - (art[0].length >> 1), top = s.y - art.length + 1;
    for (let ry = 0; ry < art.length; ry++) for (let rx = 0; rx < art[ry].length; rx++) {
      const ch = art[ry][rx]; if (ch === ".") continue;
      const xx = left + rx, yy = top + ry; if (xx < 0 || yy < 0 || xx >= W || yy >= H) continue;
      const i = yy * W + xx; if (kind[i] === 0) continue;
      const c = INK[ch];
      put(i, s.dim ? [c[0] * 0.55 + C.foreign[0] * 0.45, c[1] * 0.55 + C.foreign[1] * 0.45, c[2] * 0.55 + C.foreign[2] * 0.45] : c);
    }
  }
  for (const bt of BOATS) {
    const art = SPR[bt.dark ? "boatDark" : "boat"]; const x0 = Math.round(toX(bt.lon)), y0 = Math.round(toY(bt.lat));
    for (let ry = 0; ry < art.length; ry++) for (let rx = 0; rx < art[ry].length; rx++) {
      const ch = art[ry][rx]; if (ch === ".") continue;
      const xx = x0 + rx, yy = y0 + ry; if (xx < 0 || yy < 0 || xx >= W || yy >= H || kind[yy * W + xx] !== 0) continue;
      put(yy * W + xx, INK[ch]);
    }
  }

  const canvas = document.createElement("canvas");
  canvas.width = W; canvas.height = H;
  canvas.getContext("2d")!.putImageData(new ImageData(rgba, W, H), 0, 0);
  cached = { canvas, rgba, kind };
  return cached;
}

let cachedOutline: HTMLCanvasElement | null = null;

/** Ink outline of the art map (coast + dotted India border), drawn over the forecast when the art is dimmed (spec IMP-3g). */
export function getIndiaOutline(): HTMLCanvasElement {
  if (cachedOutline) return cachedOutline;
  const { kind } = getIndiaArt();
  const W = ART_W, H = ART_H;
  const px = new Uint8ClampedArray(W * H * 4);
  const at = (x: number, y: number) => (x < 0 || y < 0 || x >= W || y >= H ? -1 : kind[y * W + x]);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const k = kind[y * W + x];
      if (k === 0) continue;
      const n = [at(x + 1, y), at(x - 1, y), at(x, y + 1), at(x, y - 1)];
      let a = 0;
      if (n.includes(0)) a = 235;                                          // coast
      else if (k === 2 && n.includes(1) && ((x + y) & 1) === 0) a = 190;  // India border, dotted
      if (!a) continue;
      const o = (y * W + x) * 4;
      px[o] = 122; px[o + 1] = 47; px[o + 2] = 31; px[o + 3] = a;
    }
  }
  const c = document.createElement("canvas");
  c.width = W; c.height = H;
  c.getContext("2d")!.putImageData(new ImageData(px, W, H), 0, 0);
  cachedOutline = c;
  return c;
}
