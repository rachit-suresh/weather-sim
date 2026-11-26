#!/usr/bin/env node
/**
 * Builds shared/grid.json — canonical 32x36 India domain matching the actual
 * model-serving backend contract:
 *
 *   Shape: [6][32][36] = [channel][lat_row][lon_col]
 *   Row 0 = 6.5°N (south) .. Row 31 = 37.5°N (north)
 *   Col 0 = 65.5°E (west) .. Col 35 = 100.5°E (east)
 *   Channels: ["t2m", "tp", "u10", "v10", "msl", "tcwv"]
 *   Units:    ["K", "mm/day", "m/s", "m/s", "Pa", "kg/m2"]
 *
 * Mask in grid.json (local geographic reference):
 *   0 = ocean, 1 = land outside India, 2 = India land (incl. islands)
 * Note: Backend /meta provides land_sea_mask in [0, 1] (0=sea, 1=land,
 * fractional along coasts) without national borders.
 */
import { writeFileSync, mkdirSync } from "node:fs";

const NLAT = 32;
const NLON = 36;
const LAT0 = 6.5;
const LON0 = 65.5;
const STEP = 1.0;

const lat = Array.from({ length: NLAT }, (_, r) => +(LAT0 + r * STEP).toFixed(2));
const lon = Array.from({ length: NLON }, (_, c) => +(LON0 + c * STEP).toFixed(2));

// Old RLE rows were north->south (37.5N..6.5N) and lon 67.5E..102.5E.
// Shift west by 2 cols (65.5E..100.5E) and reverse rows to south->north (6.5N..37.5N).
const OLD_RLE_NORTH_TO_SOUTH = [
  [[36, 1]],                                                        // 37.5N
  [[36, 1]],                                                        // 36.5N
  [[36, 1]],                                                        // 35.5N
  [[7, 1], [4, 2], [25, 1]],                                        // 34.5N  Ladakh
  [[6, 1], [5, 2], [25, 1]],                                        // 33.5N
  [[7, 1], [5, 2], [24, 1]],                                        // 32.5N  J&K / Himachal
  [[7, 1], [6, 2], [23, 1]],                                        // 31.5N  Punjab / Uttarakhand
  [[7, 1], [7, 2], [22, 1]],                                        // 30.5N  Haryana / UP
  [[6, 1], [9, 2], [21, 1]],                                        // 29.5N  Rajasthan
  [[5, 1], [11, 2], [5, 1], [1, 2], [3, 1], [5, 2], [6, 1]],        // 28.5N  Delhi | Sikkim | Arunachal
  [[4, 1], [13, 2], [3, 1], [2, 2], [2, 1], [6, 2], [6, 1]],        // 27.5N
  [[3, 1], [20, 2], [1, 1], [6, 2], [6, 1]],                        // 26.5N  Bihar / Bengal / NE
  [[3, 1], [19, 2], [3, 1], [4, 2], [7, 1]],                        // 25.5N
  [[3, 1], [18, 2], [4, 1], [3, 2], [8, 1]],                        // 24.5N
  [[1, 0], [20, 2], [4, 1], [2, 2], [9, 1]],                        // 23.5N  Kutch
  [[2, 0], [20, 2], [14, 1]],                                       // 22.5N  Saurashtra / Kolkata
  [[3, 0], [18, 2], [1, 0], [14, 1]],                               // 21.5N
  [[5, 0], [15, 2], [6, 0], [10, 1]],                               // 20.5N
  [[5, 0], [14, 2], [7, 0], [10, 1]],                               // 19.5N  Mumbai
  [[5, 0], [13, 2], [8, 0], [10, 1]],                               // 18.5N
  [[6, 0], [11, 2], [9, 0], [10, 1]],                               // 17.5N  Hyderabad / Vizag
  [[6, 0], [10, 2], [10, 0], [10, 1]],                              // 16.5N
  [[7, 0], [7, 2], [12, 0], [10, 1]],                               // 15.5N  Goa
  [[7, 0], [7, 2], [12, 0], [10, 1]],                               // 14.5N
  [[8, 0], [6, 2], [12, 0], [10, 1]],                               // 13.5N  Chennai
  [[8, 0], [5, 2], [12, 0], [1, 2], [10, 1]],                       // 12.5N  Bengaluru / Andaman
  [[9, 0], [4, 2], [12, 0], [1, 2], [10, 1]],                       // 11.5N
  [[5, 0], [1, 2], [3, 0], [4, 2], [23, 0]],                        // 10.5N  Lakshadweep / Kochi
  [[9, 0], [3, 2], [1, 0], [1, 1], [22, 0]],                        //  9.5N  Kerala / Jaffna
  [[10, 0], [1, 2], [2, 0], [2, 1], [21, 0]],                       //  8.5N  Kanyakumari / Sri Lanka
  [[13, 0], [2, 1], [11, 0], [1, 2], [4, 0], [5, 1]],               //  7.5N  Sri Lanka / Nicobar
  [[26, 0], [1, 2], [5, 0], [4, 1]],                                //  6.5N  Nicobar / Sumatra
];

// Decode old rows then re-index to [south..north][65.5E..100.5E]
const oldMask = OLD_RLE_NORTH_TO_SOUTH.map((row) => {
  const cells = [];
  for (const [n, v] of row) for (let i = 0; i < n; i++) cells.push(v);
  if (cells.length !== NLON) throw new Error(`old row length ${cells.length}`);
  return cells;
});

const mask = Array.from({ length: NLAT }, (_, r) => {
  const rOld = NLAT - 1 - r;
  const la = lat[r];
  return Array.from({ length: NLON }, (_, c) => {
    const cOld = c - 2;
    if (cOld >= 0) return oldMask[rOld][cOld];
    // cols 0, 1 = 65.5E, 66.5E (Pakistan/Iran/Afghanistan north of 24.5N, Arabian Sea south)
    return la >= 25.5 ? 1 : 0;
  });
});

// Fractional coastal land_sea_mask (0=sea, 1=land, 0.25..0.75 along coasts) for local fallback
const landSeaMask = mask.map((row, r) =>
  row.map((v, c) => {
    const isLand = v > 0 ? 1 : 0;
    let same = 0, total = 0;
    for (let dr = -1; dr <= 1; dr++) {
      for (let dc = -1; dc <= 1; dc++) {
        if (dr === 0 && dc === 0) continue;
        const rr = r + dr, cc = c + dc;
        if (rr < 0 || rr >= NLAT || cc < 0 || cc >= NLON) continue;
        total++;
        if ((mask[rr][cc] > 0 ? 1 : 0) === isLand) same++;
      }
    }
    if (same === total) return isLand;
    return +(isLand ? 0.55 + 0.4 * (same / total) : 0.45 * (1 - same / total)).toFixed(3);
  }),
);

const idx = (r, c) => r * NLON + c;
const cellOf = (la, lo) => {
  const r = Math.round((la - LAT0) / STEP);
  const c = Math.round((lo - LON0) / STEP);
  return [Math.max(0, Math.min(NLAT - 1, r)), Math.max(0, Math.min(NLON - 1, c))];
};

const REGION_DEFS = {
  "Bay of Bengal":      [8.0, 20.0, 82.0, 94.0],
  "Arabian Sea":        [8.0, 22.0, 65.5, 73.0],
  "Western Ghats":      [8.5, 21.0, 73.0, 76.5],
  "Indo-Gangetic Plain":[24.0, 31.0, 73.0, 88.0],
  "Central India":      [17.0, 24.0, 74.0, 83.0],
  "Northeast India":    [22.0, 29.5, 89.0, 97.0],
  "Thar Desert":        [24.0, 30.0, 69.0, 74.0],
  "Deccan Plateau":     [12.0, 19.0, 75.0, 80.0],
  "Himalayan Rim":      [27.0, 36.0, 73.0, 96.0],
};

const regions = Object.entries(REGION_DEFS).map(([name, [a, b, c, d]]) => {
  const cells = [];
  for (let r = 0; r < NLAT; r++)
    for (let k = 0; k < NLON; k++)
      if (lat[r] >= a && lat[r] <= b && lon[k] >= c && lon[k] <= d) cells.push([r, k]);
  return { name, bbox: [a, b, c, d], cells };
});

const CITY_DEFS = [
  ["Mumbai", 19.08, 72.88, "west coast"],
  ["Delhi", 28.61, 77.21, "Indo-Gangetic"],
  ["Kolkata", 22.57, 88.36, "Ganges delta"],
  ["Chennai", 13.08, 80.27, "east coast"],
  ["Bengaluru", 12.97, 77.59, "Deccan"],
  ["Hyderabad", 17.39, 78.49, "Deccan"],
  ["Ahmedabad", 23.02, 72.57, "Gujarat"],
  ["Guwahati", 26.14, 91.74, "Brahmaputra"],
  ["Kochi", 9.93, 76.26, "Malabar coast"],
  ["Bhubaneswar", 20.3, 85.82, "Odisha coast"],
  ["Jaipur", 26.91, 75.79, "Thar margin"],
  ["Visakhapatnam", 17.69, 83.22, "north Andhra coast"],
  ["Patna", 25.59, 85.14, "mid-Ganges"],
  ["Nagpur", 21.15, 79.09, "central India"],
  ["Srinagar", 34.08, 74.8, "Kashmir valley"],
  ["Port Blair", 11.62, 92.73, "Andaman Sea"],
];
const cities = CITY_DEFS.map(([name, la, lo, note]) => {
  const [r, c] = cellOf(la, lo);
  return { name, lat: la, lon: lo, note, r, c, i: idx(r, c) };
});

const elevation = Array.from({ length: NLAT }, () => new Array(NLON).fill(0));
function bump(la, lo, height, sigmaLat, sigmaLon) {
  for (let r = 0; r < NLAT; r++)
    for (let c = 0; c < NLON; c++) {
      const dy = (lat[r] - la) / sigmaLat;
      const dx = (lon[c] - lo) / sigmaLon;
      elevation[r][c] = Math.min(1, elevation[r][c] + height * Math.exp(-0.5 * (dx * dx + dy * dy)));
    }
}
bump(33.0, 80.0, 0.95, 1.6, 8.0);  // Himalayan ridge
bump(35.5, 88.0, 1.0, 3.0, 12.0);  // Tibetan plateau
bump(34.5, 76.0, 0.95, 2.0, 3.0);  // Karakoram / Ladakh
bump(15.0, 74.6, 0.55, 5.0, 0.8);  // Western Ghats
bump(20.5, 80.0, 0.3, 3.0, 4.0);   // Satpura / Vindhya
bump(25.1, 91.8, 0.30, 0.9, 1.9);  // Meghalaya plateau
bump(12.5, 76.5, 0.35, 2.0, 1.5);  // Nilgiris
bump(27.0, 71.5, 0.12, 2.5, 2.5);  // Aravalli
bump(22.5, 94.0, 0.26, 2.4, 1.3);  // Chin hills

const offLand = cities.filter((c) => mask[c.r][c.c] !== 2);
if (offLand.length) {
  throw new Error("cities not on India land cells: " + offLand.map((c) => `${c.name}(r${c.r},c${c.c},mask=${mask[c.r][c.c]})`).join(", "));
}

const grid = {
  nlat: NLAT, nlon: NLON, step: STEP, lat0: LAT0, lon0: LON0,
  order: "lat_asc_lon_asc",
  lat, lon, mask, land_sea_mask: landSeaMask, regions, cities,
  elevation: elevation.map((row) => row.map((v) => +v.toFixed(4))),
  channels: ["t2m", "tp", "u10", "v10", "msl", "tcwv"],
  units: { t2m: "K", tp: "mm/day", u10: "m/s", v10: "m/s", msl: "Pa", tcwv: "kg/m2" },
  labels: {
    t2m: "2 m temperature", tp: "Total precipitation", u10: "10 m eastward wind",
    v10: "10 m northward wind", msl: "Mean sea-level pressure", tcwv: "Column water vapour",
  },
  max_steps: 14,
};

mkdirSync("shared", { recursive: true });
writeFileSync("shared/grid.json", JSON.stringify(grid));

const glyph = { 0: "·", 1: "░", 2: "█" };
console.log("32x36 India domain (north shown at top) — █ India  ░ other land  · ocean");
console.log("      " + lon.map((v) => String(Math.round(v) % 10)).join(""));
for (let r = NLAT - 1; r >= 0; r--) {
  console.log(String(lat[r]).padStart(5) + " " + mask[r].map((v) => glyph[v]).join(""));
}
const counts = { ocean: 0, foreign: 0, india: 0 };
mask.flat().forEach((v) => counts[v === 0 ? "ocean" : v === 1 ? "foreign" : "india"]++);
console.log("\ncells:", counts, "total", NLAT * NLON);
console.log("wrote shared/grid.json");
