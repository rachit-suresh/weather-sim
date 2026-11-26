/**
 * Lightweight 0.5-degree global continent & biome lookup table (720 x 360)
 * built once from simplified [lon, lat] continent polygons.
 * Used exclusively for non-India landmasses on the storybook globe;
 * the [6..38°N, 65..101°E] simulation grid renders from GRID.mask & TERRAIN.
 */

export const WORLD_NLON = 720; // 0.5 deg columns: lon = -180 + (c + 0.5) * 0.5
export const WORLD_NLAT = 360; // 0.5 deg rows:    lat = -90  + (r + 0.5) * 0.5

// Biome IDs stored in WORLD_BIOME (0 = ocean, 1..5 = land biomes matching storybook palette)
export type WorldBiome = 0 | 1 | 2 | 3 | 4 | 5;
// 0: ocean, 1: temperate grass, 2: lush forest, 3: desert/sand, 4: highland/mountain, 5: polar ice/snow

type LonLat = readonly [number, number];

// Simplified real-world landmass rings ([lon, lat] in degrees)
const LAND_POLYGONS: readonly (readonly LonLat[])[] = [
  // North America + Central America
  [
    [-168, 66], [-162, 71], [-140, 70], [-124, 70], [-110, 69], [-96, 70], [-82, 69],
    [-77, 63], [-64, 60], [-56, 52], [-60, 46], [-66, 45], [-70, 42], [-76, 35],
    [-81, 25], [-83, 29], [-89, 30], [-97, 26], [-96, 19], [-91, 18], [-87, 21],
    [-88, 16], [-84, 15], [-83, 10], [-78, 8], [-79, 9], [-85, 11], [-92, 14],
    [-105, 19], [-110, 24], [-114, 31], [-117, 32], [-123, 38], [-125, 49],
    [-133, 56], [-148, 60], [-162, 55], [-166, 60], [-168, 66],
  ],
  // Greenland
  [[-52, 60], [-42, 60], [-22, 70], [-18, 81], [-45, 83], [-70, 78], [-58, 69], [-52, 60]],
  // Baffin & Canadian Archipelago
  [[-82, 72], [-64, 67], [-67, 62], [-77, 64], [-86, 71], [-82, 72]],
  [[-120, 74], [-98, 76], [-96, 71], [-118, 70], [-120, 74]],
  // Cuba & Hispaniola
  [[-85, 22], [-74, 20], [-75, 21], [-84, 23], [-85, 22]],
  [[-74, 19], [-68, 18], [-69, 20], [-74, 20], [-74, 19]],
  // Iceland
  [[-24, 65], [-14, 66], [-14, 64], [-24, 63], [-24, 65]],

  // South America
  [
    [-78, 9], [-72, 12], [-62, 10], [-50, 1], [-35, -6], [-38, -14], [-40, -22],
    [-48, -26], [-53, -34], [-61, -39], [-65, -43], [-68, -52], [-66, -55],
    [-73, -54], [-75, -46], [-72, -35], [-71, -19], [-77, -12], [-81, -5], [-79, 2], [-78, 9],
  ],

  // UK & Ireland
  [[-6, 50], [1, 51], [1, 54], [-3, 58], [-6, 57], [-5, 51], [-6, 50]],
  [[-10, 52], [-6, 55], [-6, 52], [-9, 51], [-10, 52]],

  // Africa
  [
    [-6, 36], [10, 37], [12, 33], [24, 32], [32, 31], [34, 27], [40, 18], [43, 12],
    [51, 12], [49, 4], [41, -2], [40, -11], [35, -20], [32, -28], [27, -34],
    [18, -35], [15, -26], [12, -16], [12, -6], [9, 4], [2, 6], [-8, 5],
    [-17, 15], [-16, 23], [-10, 30], [-6, 36],
  ],
  // Madagascar
  [[44, -25], [47, -25], [50, -15], [49, -12], [44, -18], [44, -25]],

  // Eurasia (connected cleanly to India domain edge)
  [
    [-9, 37], [-9, 43], [-2, 43], [-5, 48], [2, 51], [8, 54], [9, 58], [5, 61],
    [15, 69], [28, 71], [42, 68], [58, 69], [70, 73], [92, 77], [112, 76],
    [136, 72], [160, 70], [178, 66], [164, 60], [160, 52], [142, 54], [140, 46],
    [132, 43], [129, 39], [127, 35], [129, 35], [126, 39], [122, 40], [118, 39],
    [122, 37], [120, 34], [122, 31], [120, 26], [114, 22], [108, 21], [109, 13],
    [105, 9], [100, 13], [100.5, 11], [103.5, 2], [100, 5.5], [98.5, 9.5], [97, 16],
    [94, 16], [92, 21], [89, 22], [85, 19], [80, 14], [78, 8], [76, 9], [73, 16],
    [70, 22], [67, 25], [61, 25], [56, 26], [51, 26], [49, 30], [56, 24], [59, 22],
    [54, 17], [44, 13], [39, 21], [34, 29], [32, 31], [35, 33], [36, 37], [27, 37],
    [26, 41], [23, 38], [19, 40], [14, 45], [12, 41], [16, 38], [12, 42], [9, 44],
    [3, 42], [-2, 37], [-6, 36], [-9, 37],
  ],

  // Sri Lanka
  [[79.7, 9.6], [81.0, 8.6], [81.8, 7.4], [81.2, 6.1], [80.2, 6.0], [79.7, 7.5], [79.7, 9.6]],

  // Japan
  [[130, 31], [133, 33], [137, 35], [141, 38], [142, 43], [145, 44], [141, 41], [139, 38], [135, 35], [130, 33], [130, 31]],

  // Maritime SE Asia
  [[95, 5], [100, 2], [106, -5], [103, -4], [97, 2], [95, 5]], // Sumatra
  [[105, -6], [114, -8], [114, -7], [106, -7], [105, -6]], // Java
  [[109, 1], [117, 7], [119, 1], [115, -4], [110, -3], [109, 1]], // Borneo
  [[119, -5], [125, 1], [122, -4], [119, -5]], // Sulawesi
  [[120, 14], [122, 18], [126, 8], [123, 7], [121, 13], [120, 14]], // Philippines
  [[131, -2], [141, -3], [150, -10], [144, -8], [135, -5], [131, -2]], // New Guinea

  // Australia & Tasmania & New Zealand
  [
    [114, -22], [122, -17], [131, -12], [136, -12], [137, -15], [141, -16],
    [142, -11], [146, -18], [153, -28], [150, -37], [141, -38], [135, -34],
    [129, -32], [122, -34], [115, -34], [113, -26], [114, -22],
  ],
  [[145, -41], [148, -41], [147, -43.5], [145, -42], [145, -41]],
  [[167, -46], [174, -41], [178, -37], [173, -35], [173, -41], [166, -45], [167, -46]],
];

function pointInRing(lon: number, lat: number, ring: readonly LonLat[]): boolean {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const xi = ring[i][0], yi = ring[i][1];
    const xj = ring[j][0], yj = ring[j][1];
    const intersect = (yi > lat) !== (yj > lat) &&
      lon < ((xj - xi) * (lat - yi)) / (yj - yi + 1e-12) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

function classifyWorldBiome(lat: number, lon: number): WorldBiome {
  // Antarctica
  const antEdge = -68 + 2.8 * Math.sin(lon * 0.052) + 1.8 * Math.cos(lon * 0.087);
  if (lat <= antEdge) return 5;

  let isLand = false;
  for (let p = 0; p < LAND_POLYGONS.length; p++) {
    if (pointInRing(lon, lat, LAND_POLYGONS[p])) {
      isLand = true;
      break;
    }
  }
  if (!isLand) return 0;

  // Polar ice / tundra
  if (lat >= 68 || (lat >= 60 && lon >= -60 && lon <= -18)) return 5;

  // Organic elliptical mountain belts (no axis-aligned rectangle corners)
  const inEllipse = (la: number, lo: number, clat: number, clon: number, rlat: number, rlon: number) =>
    ((la - clat) / rlat) ** 2 + ((lo - clon) / rlon) ** 2 <= 1;

  const isTibet = inEllipse(lat, lon, 33.0, 86.5, 6.2, 15.0);
  const isAndes = lat >= -45 && lat <= 10 && lon >= -78 && lon <= -65;
  const isRockies = lat >= 25 && lat <= 60 && lon >= -124 && lon <= -103;
  const isAlps = inEllipse(lat, lon, 46.2, 11.5, 2.2, 5.5);
  const isEthiopia = inEllipse(lat, lon, 10.5, 39.0, 4.5, 3.2);
  if (isTibet || isAndes || isRockies || isAlps || isEthiopia) return 4;

  // Deserts & plateaus (curved zones matching western border tan)
  const isSahara = inEllipse(lat, lon, 23.5, 11.0, 9.0, 26.0);
  const isArabiaIran = inEllipse(lat, lon, 27.0, 52.5, 12.5, 19.5);
  const isTarimGobi = inEllipse(lat, lon, 41.5, 92.0, 4.5, 19.0);
  const isOutback = inEllipse(lat, lon, -24.5, 128.5, 6.2, 13.5);
  const isSwUs = inEllipse(lat, lon, 30.5, -109.0, 6.0, 7.5);
  if (isSahara || isArabiaIran || isTarimGobi || isOutback || isSwUs) return 3;

  // Equatorial / Monsoon lush forest (Amazon, Congo, SE Asia & southern China)
  if (Math.abs(lat) <= 13 || (lat >= 13 && lat <= 28 && lon >= 90 && lon <= 112) || (lat >= 48 && lat <= 64 && lon >= 10 && lon <= 140)) {
    return 2;
  }

  return 1;
}

export const WORLD_BIOME: Uint8Array = (() => {
  const arr = new Uint8Array(WORLD_NLON * WORLD_NLAT);
  for (let r = 0; r < WORLD_NLAT; r++) {
    const lat = -90 + (r + 0.5) * 0.5;
    for (let c = 0; c < WORLD_NLON; c++) {
      const lon = -180 + (c + 0.5) * 0.5;
      arr[r * WORLD_NLON + c] = classifyWorldBiome(lat, lon);
    }
  }
  return arr;
})();

export const WORLD_RGB: Record<WorldBiome, readonly [number, number, number]> = {
  0: [43, 157, 199],  // ocean teal, matches the India art map sea (spec IMP-3)
  1: [164, 198, 114], // temperate meadow
  2: [110, 168, 86],  // lush forest
  3: [217, 196, 143], // plateau / desert sand (matches foreign India border terrain)
  4: [200, 182, 220], // alpine highland (matches Himalayan rim)
  5: [244, 240, 224], // polar ice
};

/** Wrap longitude cleanly into [-180, 180) and sample precomputed 0.5° biome map. */
export function sampleWorldBiome(latDeg: number, lonDeg: number): WorldBiome {
  let lon = ((lonDeg + 180) % 360 + 360) % 360 - 180;
  if (lon >= 180) lon -= 360;
  const lat = Math.max(-89.99, Math.min(89.99, latDeg));
  const r = Math.max(0, Math.min(WORLD_NLAT - 1, Math.floor((lat + 90) * 2)));
  const c = Math.max(0, Math.min(WORLD_NLON - 1, Math.floor((lon + 180) * 2)));
  return WORLD_BIOME[r * WORLD_NLON + c] as WorldBiome;
}

/* ════════════════════════════════════════════════════════════════════════════
   Conformal Möbius Globe Focus Warp (tan(θ'/2) = k · tan(θ/2) around anchor)
   Enlarges India smoothly across the front of the sphere while shrinking other
   continents conformally onto the rest of the globe.
   Set GLOBE_FOCUS_SCALE = 1.0 when adding more countries at true planetary scale.
   ════════════════════════════════════════════════════════════════════════════ */
const DEG = Math.PI / 180;
const RAD2DEG = 180 / Math.PI;

export const GLOBE_FOCUS_CENTER = { lat: 21.5, lon: 80.5 };
export const GLOBE_FOCUS_SCALE = 5.0; // spec IMP-3c: India comically large (was 2.65)
export const GLOBE_FOCUS_INV_SCALE = 1 / GLOBE_FOCUS_SCALE;

export function latLonToUnitVec(latDeg: number, lonDeg: number): [number, number, number] {
  const lat = latDeg * DEG;
  const lon = lonDeg * DEG;
  const cl = Math.cos(lat);
  return [cl * Math.sin(lon), Math.sin(lat), cl * Math.cos(lon)];
}

export function unitVecToLatLon(ux: number, uy: number, uz: number): { lat: number; lon: number } {
  return {
    lat: Math.asin(Math.max(-1, Math.min(1, uy))) * RAD2DEG,
    lon: Math.atan2(ux, uz) * RAD2DEG,
  };
}

export const [ANCHOR_X, ANCHOR_Y, ANCHOR_Z] = latLonToUnitVec(
  GLOBE_FOCUS_CENTER.lat,
  GLOBE_FOCUS_CENTER.lon,
);

/**
 * Conformal stereographic radial scale about ANCHOR:
 * Pass GLOBE_FOCUS_SCALE for geo -> display, or GLOBE_FOCUS_INV_SCALE for display -> geo.
 */
export function mobiusWarpVec(
  ux: number,
  uy: number,
  uz: number,
  k: number,
): { x: number; y: number; z: number; localScale: number } {
  if (k === 1) return { x: ux, y: uy, z: uz, localScale: 1 };
  const c = ux * ANCHOR_X + uy * ANCHOR_Y + uz * ANCHOR_Z;
  const k2 = k * k;
  const onePlusC = 1 + c;
  const oneMinusC = 1 - c;
  const denom = onePlusC + k2 * oneMinusC;
  const a = (onePlusC - k2 * oneMinusC) / denom;
  const b = (2 * k) / denom;
  return {
    x: a * ANCHOR_X + b * (ux - c * ANCHOR_X),
    y: a * ANCHOR_Y + b * (uy - c * ANCHOR_Y),
    z: a * ANCHOR_Z + b * (uz - c * ANCHOR_Z),
    localScale: b,
  };
}

/** Convert geographic (lat, lon) to warped display-sphere (lat, lon) for camera flyTo. */
export function geoToDisplayLatLon(latDeg: number, lonDeg: number): { lat: number; lon: number } {
  const [ux, uy, uz] = latLonToUnitVec(latDeg, lonDeg);
  const w = mobiusWarpVec(ux, uy, uz, GLOBE_FOCUS_SCALE);
  return unitVecToLatLon(w.x, w.y, w.z);
}

/** Right-handed orthographic camera axes in 3D sphere space (East=right, North=up, Camera=forward). */
export function cameraAxes(camLatDeg: number, camLonDeg: number) {
  const lat = camLatDeg * DEG;
  const lon = camLonDeg * DEG;
  const sinLat = Math.sin(lat), cosLat = Math.cos(lat);
  const sinLon = Math.sin(lon), cosLon = Math.cos(lon);
  return {
    fx: cosLat * sinLon, fy: sinLat, fz: cosLat * cosLon,
    rx: cosLon, ry: 0, rz: -sinLon,
    upx: -sinLat * sinLon, upy: cosLat, upz: -sinLat * cosLon,
  };
}
