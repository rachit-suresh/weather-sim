import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { animate, createTimer, stagger } from "animejs";
import { motion, AnimatePresence, useReducedMotion } from "motion/react";
import confetti from "canvas-confetti";
import {
  Plus, Minus, Compass, Shuffle, ChevronLeft, ChevronRight, Radio,
  RotateCw, Maximize2, Crosshair,
} from "lucide-react";
import {
  CELLS, GRID, NC, NR, colOf, rowOf, idx, inBounds,
  fmtLat, fmtLon, regionOf, nearestCity, display, displayDelta, displayUnit,
  windMetrics, type VarKey,
} from "@/lib/grid";
import { frame, diffFrame, type Timeline } from "@/lib/api";
import {
  PAPER, TERRAIN, cellHash, stamp, computeTargetField,
  colorForFieldValue, brushCellsAround, drawPixelSun, drawCompassRose, drawPixelSonarRipple, storyAlpha,
} from "@/lib/mapSprites";
import { getIndiaArt, artIndex } from "@/lib/indiaArt";
import {
  WORLD_RGB, sampleWorldBiome,
  GLOBE_FOCUS_SCALE, GLOBE_FOCUS_INV_SCALE,
  latLonToUnitVec, unitVecToLatLon, mobiusWarpVec, geoToDisplayLatLon, cameraAxes,
} from "@/lib/worldMask";
import type { LensId, LensOutput } from "@/lib/impact";
import type { ViewMode, Ripple } from "@/components/PixelGrid";

/* ════════════════════════════════════════════════════════════════════════════
   GlobeView — Cartoony Pixel-Art Radio-Garden Globe for India's 32×36 Rollout
   - Quantized inverse-orthographic pixel sphere shader + maroon coastlines
   - Storybook sprites (trees, pines, peaks, cacti, ships, clouds, sun, birds)
   - Radio-Garden tuning reticle + live station beacons + smooth fly-to flights
   ════════════════════════════════════════════════════════════════════════════ */

interface Props {
  timeline: Timeline | null;
  landSeaMask: number[][];
  view: ViewMode;
  variable: VarKey;
  day: number;
  lens: LensId | null;
  lensOut: LensOutput | null;
  paintMask: Map<number, number>;
  paintMode: boolean;
  brushRadius: number;
  onPaint: (cells: { i: number; w: number }[], erase: boolean) => void;
  onPaintEnd: () => void;
  selected: number | null;
  onSelect: (i: number | null) => void;
  ripples: Ripple[];
  showWind: boolean;
  showCities: boolean;
  showScenery: boolean;
  weatherFocus?: number;
  running: boolean;
  focusRegion: string | null;
  onSwitchToFlatMap: () => void;
}

const DEG = Math.PI / 180;
const RAD2DEG = 180 / Math.PI;
const INDIA_HOME = { lat: 21.5, lon: 80.5, zoom: 1.04 };
const INITIAL_TUNED_IDX = Math.max(0, GRID.cities.findIndex((c) => c.name === "Nagpur"));
const MIN_ZOOM = 0.78;
const MAX_ZOOM = 2.8;
const PIXEL_STEP = 3; // chunky screen-pixel size for authentic cartoony pixel-art sphere

// Pack RGBA into 32-bit little-endian ImageData int
const packRgba = (r: number, g: number, b: number, a = 255) =>
  ((a & 255) << 24) | ((b & 255) << 16) | ((g & 255) << 8) | (r & 255);

const INK_SHORE_U32 = packRgba(122, 47, 31, 230);
const INK_RIM_U32 = packRgba(105, 68, 46, 255);
const INK_BORDER_U32 = packRgba(122, 47, 31, 125);

const normalizeLon = (lon: number) => (((lon + 180) % 360) + 360) % 360 - 180;
const shortestLonDelta = (fromLon: number, toLon: number) => normalizeLon(toLon - fromLon);

interface Projected {
  x: number;
  y: number;
  cosC: number;
  localScale: number;
  visible: boolean;
}

type CamAxes = ReturnType<typeof cameraAxes>;

function projectGeo(
  geoLat: number,
  geoLon: number,
  axes: CamAxes,
  R: number,
  cx: number,
  cy: number,
  minCosC = 0.02,
): Projected {
  const [ux, uy, uz] = latLonToUnitVec(geoLat, geoLon);
  const w = mobiusWarpVec(ux, uy, uz, GLOBE_FOCUS_SCALE);
  const nx = w.x * axes.rx + w.y * axes.ry + w.z * axes.rz;
  const ny = w.x * axes.upx + w.y * axes.upy + w.z * axes.upz;
  const cosC = w.x * axes.fx + w.y * axes.fy + w.z * axes.fz;
  return {
    x: cx + R * nx,
    y: cy - R * ny,
    cosC,
    localScale: w.localScale,
    visible: cosC >= minCosC,
  };
}

function unprojectGeo(
  sx: number,
  sy: number,
  axes: CamAxes,
  R: number,
  cx: number,
  cy: number,
): { lat: number; lon: number; r2: number } | null {
  const nx = (sx - cx) / R;
  const ny = (cy - sy) / R;
  const r2 = nx * nx + ny * ny;
  if (r2 > 1) return null;
  const cosC = Math.sqrt(Math.max(0, 1 - r2));
  const dx = nx * axes.rx + ny * axes.upx + cosC * axes.fx;
  const dy = nx * axes.ry + ny * axes.upy + cosC * axes.fy;
  const dz = nx * axes.rz + ny * axes.upz + cosC * axes.fz;
  const u = mobiusWarpVec(dx, dy, dz, GLOBE_FOCUS_INV_SCALE);
  const geo = unitVecToLatLon(u.x, u.y, u.z);
  return { lat: geo.lat, lon: normalizeLon(geo.lon), r2 };
}

export default function GlobeView(props: Props) {
  const reduceMotion = useReducedMotion();
  const {
    timeline, landSeaMask, view, variable, day, lens, lensOut, paintMask, paintMode, brushRadius,
    onPaint, onPaintEnd, selected, onSelect, ripples, showWind, showCities, running, focusRegion,
    onSwitchToFlatMap,
  } = props;

  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const celebrationRef = useRef<HTMLCanvasElement>(null);
  const cannonRef = useRef<confetti.CreateTypes | null>(null);
  const bufCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const [celebrating, setCelebrating] = useState(false);
  const [hover, setHover] = useState<number | null>(null);
  const [cursorPx, setCursorPx] = useState<{ x: number; y: number } | null>(null);
  const [tunedIdx, setTunedIdx] = useState<number>(INITIAL_TUNED_IDX); // index in GRID.cities
  const [autoSpin, setAutoSpin] = useState(false);
  const [size, setSize] = useState({ w: 0, h: 0 });

  // Globe camera & interaction refs
  const camRef = useRef({ lat: INDIA_HOME.lat, lon: INDIA_HOME.lon, zoom: INDIA_HOME.zoom });
  const velRef = useRef({ lat: 0, lon: 0 });
  const flightAnimRef = useRef<ReturnType<typeof animate> | null>(null);
  const dragRef = useRef({
    active: false,
    mode: "spin" as "spin" | "paint",
    startX: 0,
    startY: 0,
    lastX: 0,
    lastY: 0,
    startLat: INDIA_HOME.lat,
    startLon: INDIA_HOME.lon,
    movedDist: 0,
    erase: false,
    lastCell: null as number | null,
  });

  // Animated weather display state
  const fromRef = useRef(new Float32Array(CELLS));
  const toRef = useRef(new Float32Array(CELLS));
  const dispRef = useRef(new Float32Array(CELLS));
  const blendRef = useRef({ v: 1 });
  const scaleRef = useRef({ from: 1, to: 1, v: 1 });
  const selPulse = useRef({ v: 0 });
  const ringRef = useRef<{ lat: number; lon: number; p: number; kind: Ripple["kind"]; k: number }[]>([]);
  const seenRipplesRef = useRef<Set<number> | null>(null);
  const activeRippleAnimsRef = useRef(new Set<ReturnType<typeof animate>>());
  const sweepRef = useRef({ v: 1 });

  const propsRef = useRef(props);
  propsRef.current = props;
  const hoverRef = useRef<number | null>(null);
  hoverRef.current = hover;
  const autoSpinRef = useRef(autoSpin);
  autoSpinRef.current = autoSpin;
  const tunedIdxRef = useRef(tunedIdx);
  tunedIdxRef.current = tunedIdx;

  // Compute globe pixel geometry helper from canvas dimensions & zoom
  const globeGeom = useCallback((w: number, h: number, zoom: number) => {
    const baseR = Math.max(85, Math.min(w, h) * 0.41);
    return { R: baseR * zoom, baseR, cx: w / 2, cy: h / 2 };
  }, []);

  /* ── smoothly animate camera to center a geographic (lat, lon) and zoom ── */
  const flyTo = useCallback((geoLat: number, geoLon: number, targetZoom?: number, duration = 700) => {
    flightAnimRef.current?.pause();
    velRef.current = { lat: 0, lon: 0 };
    const targetDisp = geoToDisplayLatLon(geoLat, geoLon);
    const startLat = camRef.current.lat;
    const startLon = normalizeLon(camRef.current.lon);
    const dLon = shortestLonDelta(startLon, targetDisp.lon);
    const goalZoom = targetZoom ?? camRef.current.zoom;
    if (reduceMotion) {
      camRef.current = { lat: targetDisp.lat, lon: normalizeLon(targetDisp.lon), zoom: goalZoom };
      return;
    }
    const proxy = { t: 0 };
    const startZoom = camRef.current.zoom;
    flightAnimRef.current = animate(proxy, {
      t: 1,
      duration,
      ease: "outCubic",
      onUpdate: () => {
        camRef.current.lat = startLat + (targetDisp.lat - startLat) * proxy.t;
        camRef.current.lon = normalizeLon(startLon + dLon * proxy.t);
        camRef.current.zoom = startZoom + (goalZoom - startZoom) * proxy.t;
      },
    });
  }, [reduceMotion]);

  const tuneCity = useCallback((cityIndex: number, alsoSelect = true) => {
    const wrapped = ((cityIndex % GRID.cities.length) + GRID.cities.length) % GRID.cities.length;
    const city = GRID.cities[wrapped];
    setTunedIdx(wrapped);
    setAutoSpin(false);
    flyTo(city.lat, city.lon, Math.max(camRef.current.zoom, 1.24), 680);
    if (alsoSelect) onSelect(city.i);
  }, [flyTo, onSelect]);

  /* ── auto-orient when paintMode activates, focusRegion changes, or cell is selected externally ── */
  const prevPaintRef = useRef(paintMode);
  useEffect(() => {
    if (paintMode && !prevPaintRef.current) {
      setAutoSpin(false);
      const { lat, lon, zoom } = camRef.current;
      const farFromIndia = Math.abs(lat - INDIA_HOME.lat) > 25 || Math.abs(shortestLonDelta(lon, INDIA_HOME.lon)) > 28;
      if (zoom < 1.1 || farFromIndia) {
        flyTo(INDIA_HOME.lat, INDIA_HOME.lon, Math.max(zoom, 1.22), 620);
      }
    }
    prevPaintRef.current = paintMode;
  }, [paintMode, flyTo]);

  useEffect(() => {
    if (!focusRegion) return;
    const reg = GRID.regions.find((x) => x.name === focusRegion);
    if (!reg) return;
    const [a, b, x, y] = reg.bbox;
    setAutoSpin(false);
    flyTo((a + b) / 2, (x + y) / 2, 1.42, 650);
  }, [focusRegion, flyTo]);

  useEffect(() => {
    if (selected === null) return;
    const r = rowOf(selected), c = colOf(selected);
    const lat = GRID.lat[r], lon = GRID.lon[c];
    const nc = nearestCity(r, c);
    if (nc && nc.distanceCells <= 1.6) {
      const ci = GRID.cities.findIndex((ct) => ct.name === nc.city.name);
      if (ci >= 0) setTunedIdx(ci);
    }
    const dispSel = geoToDisplayLatLon(lat, lon);
    const dLon = Math.abs(shortestLonDelta(camRef.current.lon, dispSel.lon));
    const dLat = Math.abs(camRef.current.lat - dispSel.lat);
    if (dLon > 42 || dLat > 42) {
      setAutoSpin(false);
      flyTo(lat, lon, Math.max(camRef.current.zoom, 1.22), 620);
    }
  }, [selected, flyTo]);

  /* ── target weather field tweening ── */
  const target = useMemo(
    () => computeTargetField(timeline, view, variable, day, lens, lensOut),
    [timeline, view, variable, day, lens, lensOut],
  );

  useEffect(() => {
    fromRef.current.set(dispRef.current);
    toRef.current.set(target.values);
    scaleRef.current.from = scaleRef.current.v;
    scaleRef.current.to = target.scale;
    if (reduceMotion) {
      blendRef.current.v = 1;
      scaleRef.current.v = target.scale;
      return;
    }
    blendRef.current.v = 0;
    const a = animate(blendRef.current, { v: 1, duration: 420, ease: "outCubic" });
    const b = animate(scaleRef.current, { v: target.scale, duration: 420, ease: "outCubic" });
    return () => { a.pause(); b.pause(); };
  }, [target, reduceMotion]);

  /* ── confetti celebration when new timeline branches ── */
  useEffect(() => {
    if (!celebrationRef.current) return;
    const cannon = confetti.create(celebrationRef.current, { resize: true, disableForReducedMotion: true });
    cannonRef.current = cannon;
    return () => { cannon.reset(); cannonRef.current = null; };
  }, []);

  useEffect(() => {
    if (!timeline || !props.showScenery || reduceMotion) return;
    const diverged = timeline.branched.tp.some((v, i) => Math.abs(v - timeline.prime.tp[i]) > 1e-5)
      || timeline.branched.t2m.some((v, i) => Math.abs(v - timeline.prime.t2m[i]) > 1e-5);
    if (!diverged) return;
    setCelebrating(true);
    void cannonRef.current?.({
      particleCount: 48, spread: 85, startVelocity: 24, gravity: 0.8, ticks: 95,
      flat: true, shapes: ["square"],
      colors: ["#efb52f", "#ce4d34", "#438238", "#529db5"],
      origin: { x: 0.5, y: 0.65 },
    });
    const timeout = window.setTimeout(() => setCelebrating(false), 2000);
    return () => { window.clearTimeout(timeout); setCelebrating(false); };
  }, [timeline?.runId, props.showScenery, reduceMotion]);

  useEffect(() => {
    if (!timeline || reduceMotion) { sweepRef.current.v = 1; return; }
    sweepRef.current.v = 0;
    const a = animate(sweepRef.current, { v: 1, duration: 900, ease: "inOutQuad" });
    return () => { a.pause(); };
  }, [timeline?.runId, reduceMotion]);

  /* ── shock & click sonar ripples (non-cancelling across rapid clicks!) ── */
  const spawnGeoRipple = useCallback((lat: number, lon: number, kind: Ripple["kind"]) => {
    if (reduceMotion) return;
    const objs = [0, 1, 2].map((k) => ({ lat, lon, p: 0, kind, k }));
    ringRef.current.push(...objs);
    const anim = animate(objs, {
      p: 1,
      duration: kind === "shock" ? 1300 : 1000,
      delay: stagger(140),
      ease: "outQuart",
      onComplete: () => {
        ringRef.current = ringRef.current.filter((o) => !objs.includes(o));
        activeRippleAnimsRef.current.delete(anim);
      },
    });
    activeRippleAnimsRef.current.add(anim);
  }, [reduceMotion]);

  useEffect(() => {
    if (seenRipplesRef.current === null) {
      seenRipplesRef.current = new Set(ripples.map((r) => r.id));
      return;
    }
    for (const r of ripples) {
      if (seenRipplesRef.current.has(r.id)) continue;
      seenRipplesRef.current.add(r.id);
      if (inBounds(r.r, r.c)) {
        spawnGeoRipple(GRID.lat[r.r], GRID.lon[r.c], r.kind);
      }
    }
  }, [ripples, spawnGeoRipple]);

  useEffect(() => {
    return () => {
      for (const a of activeRippleAnimsRef.current) a.pause();
      activeRippleAnimsRef.current.clear();
      ringRef.current = [];
    };
  }, []);

  useEffect(() => {
    if (selected === null || reduceMotion) { selPulse.current.v = 0; return; }
    const t = createTimer({
      duration: 1600, loop: true,
      onUpdate: (self) => { selPulse.current.v = 0.5 + 0.5 * Math.sin((self.currentTime / 1600) * Math.PI * 2); },
    });
    return () => { t.pause(); };
  }, [selected, reduceMotion]);

  /* ── container resize observer ── */
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      const rect = el.getBoundingClientRect();
      setSize({ w: Math.round(rect.width), h: Math.round(rect.height) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  /* ── main 60fps render loop (pixel-shader sphere + crisp storybook overlay) ── */
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || size.w === 0 || size.h === 0) return;
    const ctx = canvas.getContext("2d", { alpha: true });
    if (!ctx) return;

    const dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = Math.round(size.w * dpr);
    canvas.height = Math.round(size.h * dpr);
    canvas.style.width = `${size.w}px`;
    canvas.style.height = `${size.h}px`;

    // Offscreen low-res canvas & typed buffers for chunky pixel sphere
    const bufW = Math.max(1, Math.ceil(size.w / PIXEL_STEP));
    const bufH = Math.max(1, Math.ceil(size.h / PIXEL_STEP));
    let bufCanvas = bufCanvasRef.current;
    if (!bufCanvas) {
      bufCanvas = document.createElement("canvas");
      bufCanvasRef.current = bufCanvas;
    }
    bufCanvas.width = bufW;
    bufCanvas.height = bufH;
    const bufCtx = bufCanvas.getContext("2d");
    if (!bufCtx) return;

    const imgData = bufCtx.createImageData(bufW, bufH);
    const u32 = new Uint32Array(imgData.data.buffer);
    // landFlags per quantized pixel: -1 = outside sphere, 0 = ocean, 1 = foreign/world land, 2 = India land
    const landFlags = new Int8Array(bufW * bufH);
    // spec IMP-3f: forecast cell index per pixel (-1 = outside the grid) and grid-line weight (edge melt), for the cell grid
    const cellBuf = new Int16Array(bufW * bufH);
    const gridW = new Uint8Array(bufW * bufH);

    // Pre-seed celestial chart stars (deterministic)
    const stars: { x: number; y: number; s: number; phase: number; cross: boolean }[] = [];
    for (let i = 0; i < 64; i++) {
      const h1 = ((i * 2654435761) >>> 0) / 4294967295;
      const h2 = (((i + 17) * 2246822519) >>> 0) / 4294967295;
      stars.push({
        x: h1,
        y: h2,
        s: i % 5 === 0 ? 3 : 2,
        phase: (i * 1.37) % (Math.PI * 2),
        cross: i % 9 === 0,
      });
    }

    // Wind particles inside [0..NR) x [0..NC)
    const particles: { r: number; c: number; life: number }[] = [];
    if (showWind) {
      for (let k = 0; k < 150; k++) {
        particles.push({ r: Math.random() * NR, c: Math.random() * NC, life: Math.random() });
      }
    }

    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
    let raf = 0;

    const draw = () => {
      const now = performance.now();
      const tick = reducedMotion.matches ? 0 : now * 0.06;
      const p = propsRef.current;
      const { w, h } = size;
      const cam = camRef.current;

      // Apply inertial coasting or gentle idle turntable spin
      if (!dragRef.current.active) {
        if (Math.abs(velRef.current.lat) > 0.002 || Math.abs(velRef.current.lon) > 0.002) {
          cam.lat = Math.max(-75, Math.min(75, cam.lat + velRef.current.lat));
          cam.lon = normalizeLon(cam.lon + velRef.current.lon);
          velRef.current.lat *= 0.94;
          velRef.current.lon *= 0.94;
        } else if (autoSpinRef.current && !reducedMotion.matches) {
          cam.lon = normalizeLon(cam.lon + 0.18);
        }
      }

      // Tween weather scalars
      const blend = blendRef.current.v;
      const scaleV = scaleRef.current.v;
      const from = fromRef.current, to = toRef.current, disp = dispRef.current;
      for (let i = 0; i < CELLS; i++) disp[i] = from[i] + (to[i] - from[i]) * blend;

      let focusSet: Set<number> | null = null;
      if (p.focusRegion) {
        const reg = GRID.regions.find((x) => x.name === p.focusRegion);
        if (reg) focusSet = new Set(reg.cells.map(([r, c]) => idx(r, c)));
      }

      const { R, cx, cy } = globeGeom(w, h, cam.zoom);
      const axes = cameraAxes(cam.lat, cam.lon);
      const indiaPxPerDeg = R * DEG * GLOBE_FOCUS_SCALE;
      const strong = p.view === "diff";
      const sw = sweepRef.current.v;
      const frontCol = sw < 1 ? sw * (NC + 6) - 3 : -99;

      // Bounding box in pixel buffer coordinates
      const minPx = Math.max(0, Math.floor((cx - R - PIXEL_STEP) / PIXEL_STEP));
      const maxPx = Math.min(bufW - 1, Math.ceil((cx + R + PIXEL_STEP) / PIXEL_STEP));
      const minPy = Math.max(0, Math.floor((cy - R - PIXEL_STEP) / PIXEL_STEP));
      const maxPy = Math.min(bufH - 1, Math.ceil((cy + R + PIXEL_STEP) / PIXEL_STEP));

      u32.fill(0);
      landFlags.fill(-1);
      cellBuf.fill(-1);

      const pulseSin = 0.55 + 0.45 * Math.sin(tick * 0.08);
      // Storybook mode (spec IMP-3c): India grid pixels come from the pixel-art India map
      const story = p.showScenery;
      const art = story ? getIndiaArt() : null;
      const showGridJoints = indiaPxPerDeg >= 9.0 && !story;
      // spec IMP-3f/g: faint cell grid over the art (one buffer pixel wide) and the weather-focus dimming
      const wf = story ? Math.min(1, Math.max(0, p.weatherFocus ?? 0)) : 0;
      const storyGrid = story && indiaPxPerDeg >= 4 * PIXEL_STEP;
      const gridDark = 0.9 - 0.06 * wf;

      // ── Pass 1: Conformal Möbius + Orthographic pixel-art sphere shader ───
      for (let py = minPy; py <= maxPy; py++) {
        const sy = py * PIXEL_STEP + PIXEL_STEP * 0.5;
        const ny = (cy - sy) / R;
        const ny2 = ny * ny;
        const rowOff = py * bufW;

        for (let px = minPx; px <= maxPx; px++) {
          const sx = px * PIXEL_STEP + PIXEL_STEP * 0.5;
          const nx = (sx - cx) / R;
          const r2 = nx * nx + ny2;
          if (r2 > 1) continue;

          const bi = rowOff + px;
          const cosC = Math.sqrt(Math.max(0, 1 - r2));
          // Reconstruct display-sphere unit vector from camera frame
          const dx = nx * axes.rx + ny * axes.upx + cosC * axes.fx;
          const dy = nx * axes.ry + ny * axes.upy + cosC * axes.fy;
          const dz = nx * axes.rz + ny * axes.upz + cosC * axes.fz;
          // Unwarp conformally back to geographic Earth coordinates
          const uGeo = mobiusWarpVec(dx, dy, dz, GLOBE_FOCUS_INV_SCALE);
          const latDeg = Math.asin(Math.max(-1, Math.min(1, uGeo.y))) * RAD2DEG;
          const lonDeg = normalizeLon(Math.atan2(uGeo.x, uGeo.z) * RAD2DEG);

          // 3-step cartoony cel shading with subtle Bayer dither along terminator
          const ndotl = nx * 0.42 + ny * 0.36 + cosC * 0.83;
          const dither = ((px ^ py) & 1) === 0 ? 0.035 : -0.035;
          const nd = ndotl + dither;
          let shade = 1.0;
          if (nd > 0.84) shade = 1.05;
          else if (nd < 0.24) shade = 0.78;
          else if (nd < 0.50) shade = 0.90;

          // Subtle limb glow inside ocean rim
          const rimHighlight = r2 > 0.91 && ndotl > 0.42;

          // Check whether inside [6..38°N, 65..101°E] surrogate grid
          if (latDeg >= 6.0 && latDeg < 38.0 && lonDeg >= 65.0 && lonDeg < 101.0) {
            const rr = Math.floor(latDeg - 6.0);
            const cc = Math.floor(lonDeg - 65.0);
            const ci = rr * NC + cc;
            const mask = GRID.mask[rr][cc];
            const lsm = p.landSeaMask?.[rr]?.[cc] ?? GRID.land_sea_mask[rr][cc];
            const t = TERRAIN[ci];

            let rCol = t.base[0];
            let gCol = t.base[1];
            let bCol = t.base[2];
            let artKind = -1;
            if (art) {
              const ai = artIndex(latDeg, lonDeg);
              artKind = art.kind[ai];
              rCol = art.rgba[ai * 4]; gCol = art.rgba[ai * 4 + 1]; bCol = art.rgba[ai * 4 + 2];
              // melt the rectangular art map into the world globe near the grid edge (India itself never fades)
              let f = 1;
              if (artKind !== 2) {
                const edgeDeg = Math.min(latDeg - 6.0, 38.0 - latDeg, lonDeg - 65.0, 101.0 - lonDeg);
                f = Math.max(0, Math.min(1, edgeDeg / 3.5));
                const [wr, wg, wbCol] = WORLD_RGB[sampleWorldBiome(latDeg, lonDeg)];
                rCol = wr + (rCol - wr) * f; gCol = wg + (gCol - wg) * f; bCol = wbCol + (bCol - wbCol) * f;
              }
              if (wf > 0) {
                const d = 0.6 * wf * f; // fade the art toward paper; follows the edge melt so no box edge appears
                rCol += (PAPER[0] - rCol) * d; gCol += (PAPER[1] - gCol) * d; bCol += (PAPER[2] - bCol) * d;
              }
            } else if (mask === 1) {
              const wb = sampleWorldBiome(latDeg, lonDeg);
              const [wr, wg, wbCol] = WORLD_RGB[wb];
              rCol = rCol * 0.45 + wr * 0.55;
              gCol = gCol * 0.45 + wg * 0.55;
              bCol = bCol * 0.45 + wbCol * 0.55;
            } else if (mask === 2) {
              const b = 1 + 0.03 * Math.sin(tick * 0.02 + (rr + cc) * 0.35);
              rCol *= b; gCol *= b; bCol *= b;
            }

            if (p.timeline) {
              const rawVal = disp[ci];
              const col = colorForFieldValue(rawVal, scaleV, p.view, p.variable, p.lens, p.lensOut);
              // Wide feather (5.5 deg) for non-India cells so the 32x36 grid melts naturally into Earth
              const edgeDeg = Math.min(latDeg - 6.0, 38.0 - latDeg, lonDeg - 65.0, 101.0 - lonDeg);
              const edgeFade = mask === 2 ? 1 : Math.max(0, Math.min(1, edgeDeg / 5.5));
              let a = !p.showScenery
                ? 1
                : story
                  ? storyAlpha(rawVal, scaleV, p.view, p.variable, p.lens, artKind < 0 ? mask : artKind)
                : mask === 2
                  ? (strong ? 0.92 : 0.62)
                  : mask === 0
                    ? (strong ? 0.72 : 0.11)
                    : (strong ? 0.52 : 0.14);
              if (strong && mask !== 2) {
                const mag = Math.min(1, Math.abs(rawVal) / (scaleV || 1));
                a *= mag <= 0.04 ? 0 : Math.min(1, (mag - 0.04) * 3.0);
              }
              if (wf > 0) a += (0.85 - a) * wf;
              a *= edgeFade;
              rCol = rCol * (1 - a) + col[0] * a;
              gCol = gCol * (1 - a) + col[1] * a;
              bCol = bCol * (1 - a) + col[2] * a;
            }

            // Faint graticule ticks on non-India open ocean too (seamless globe lines)
            if (mask === 0) {
              const gLat = Math.abs(((latDeg + 90) % 15) - 7.5);
              const gLon = Math.abs(((lonDeg + 180) % 15) - 7.5);
              if ((gLat > 7.25 || gLon > 7.25) && ((px + py) & 3) === 0) {
                rCol = rCol * 0.86 + 122 * 0.14;
                gCol = gCol * 0.86 + 47 * 0.14;
                bCol = bCol * 0.86 + 31 * 0.14;
              }
            }

            // Crisp 1° cell bevel lines when zoomed into India
            if (showGridJoints && p.showScenery && !strong) {
              const fx = lonDeg - (65.0 + cc);
              const fy = latDeg - (6.0 + rr);
              if (fy > 0.88) {
                rCol += (255 - rCol) * 0.14;
                gCol += (248 - gCol) * 0.14;
                bCol += (231 - bCol) * 0.14;
              } else if (fy < 0.10 || fx < 0.10) {
                rCol *= 0.92; gCol *= 0.92; bCol *= 0.92;
              }
            }

            if (storyGrid) {
              cellBuf[bi] = ci;
              const edgeDeg = Math.min(latDeg - 6.0, 38.0 - latDeg, lonDeg - 65.0, 101.0 - lonDeg);
              gridW[bi] = mask === 2 ? 255 : Math.round(255 * Math.max(0, Math.min(1, edgeDeg / 3.5)));
            }

            if (focusSet && !focusSet.has(ci)) {
              const wash = story ? 0.35 : 0.74;     // lighter over the art map (spec IMP-3)
              rCol = rCol * (1 - wash) + PAPER[0] * wash;
              gCol = gCol * (1 - wash) + PAPER[1] * wash;
              bCol = bCol * (1 - wash) + PAPER[2] * wash;
            }

            if (frontCol > -90) {
              const d = Math.abs(cc - frontCol);
              if (d < 2.2) {
                const sa = 0.5 * (1 - d / 2.2);
                rCol = rCol * (1 - sa) + 255 * sa;
                gCol = gCol * (1 - sa) + 252 * sa;
                bCol = bCol * (1 - sa) + 235 * sa;
              }
            }

            // Painted shock highlights
            const pw = p.paintMask.get(ci);
            if (pw !== undefined) {
              const pa = 0.22 + 0.45 * pw;
              rCol = rCol * (1 - pa) + 197 * pa;
              gCol = gCol * (1 - pa) + 61 * pa;
              bCol = bCol * (1 - pa) + 46 * pa;
              const edgeLat = Math.abs(latDeg - (6.5 + rr));
              const edgeLon = Math.abs(lonDeg - (65.5 + cc));
              if (edgeLat > 0.36 || edgeLon > 0.36) {
                const ga = 0.35 + 0.45 * pw * pulseSin;
                rCol = rCol * (1 - ga) + 232 * ga;
                gCol = gCol * (1 - ga) + 161 * ga;
                bCol = bCol * (1 - ga) + 12 * ga;
              }
            }

            rCol *= shade; gCol *= shade; bCol *= shade;
            if (rimHighlight) {
              rCol = rCol * 0.8 + 255 * 0.2;
              gCol = gCol * 0.8 + 250 * 0.2;
              bCol = bCol * 0.8 + 236 * 0.2;
            }

            u32[bi] = packRgba(
              Math.max(0, Math.min(255, rCol | 0)),
              Math.max(0, Math.min(255, gCol | 0)),
              Math.max(0, Math.min(255, bCol | 0)),
              255,
            );
            landFlags[bi] = artKind >= 0 ? artKind : mask === 2 ? 2 : (mask === 1 || lsm >= 0.45) ? 1 : 0;
          } else {
            // Outside India grid -> global pixel earth
            const wb = sampleWorldBiome(latDeg, lonDeg);
            const [br, bg, bb] = WORLD_RGB[wb];
            const grain = ((((Math.floor(latDeg * 2) * 73856093) ^ (Math.floor(lonDeg * 2) * 19349663)) >>> 0) % 4) - 1.5;
            let k = shade * (1 + grain * 0.022);

            let rCol = br * k;
            let gCol = bg * k;
            let bCol = bb * k;

            // Faint dotted lat/lon graticule every 15 degrees
            const gLat = Math.abs(((latDeg + 90) % 15) - 7.5);
            const gLon = Math.abs(((lonDeg + 180) % 15) - 7.5);
            if ((gLat > 7.25 || gLon > 7.25) && ((px + py) & 3) === 0) {
              rCol = rCol * 0.86 + 122 * 0.14;
              gCol = gCol * 0.86 + 47 * 0.14;
              bCol = bCol * 0.86 + 31 * 0.14;
            }

            // Ocean wave sparkle flecks outside India
            if (wb === 0 && p.showScenery) {
              const wh = ((Math.floor(latDeg * 1.4) * 92837111) ^ (Math.floor(lonDeg * 1.4) * 689287499)) >>> 0;
              if (wh % 19 === 0 && ((px ^ py) & 1) === 0) {
                rCol = rCol * 0.68 + 251 * 0.32;
                gCol = gCol * 0.68 + 245 * 0.32;
                bCol = bCol * 0.68 + 228 * 0.32;
              }
            }

            if (rimHighlight) {
              rCol = rCol * 0.78 + 255 * 0.22;
              gCol = gCol * 0.78 + 252 * 0.22;
              bCol = bCol * 0.78 + 240 * 0.22;
            }

            u32[bi] = packRgba(
              Math.max(0, Math.min(255, rCol | 0)),
              Math.max(0, Math.min(255, gCol | 0)),
              Math.max(0, Math.min(255, bCol | 0)),
              255,
            );
            landFlags[bi] = wb === 0 ? 0 : 1;
          }
        }
      }

      // ── Pass 1b (spec IMP-3f): faint 1-pixel cell grid where the forecast cell changes from the left/upper pixel ─
      if (storyGrid) {
        for (let py = Math.max(1, minPy); py <= maxPy; py++) {
          const rowOff = py * bufW;
          for (let px = Math.max(1, minPx); px <= maxPx; px++) {
            const bi = rowOff + px;
            const ci = cellBuf[bi];
            if (ci < 0) continue;
            const lf = cellBuf[bi - 1], up = cellBuf[bi - bufW];
            if ((lf >= 0 && lf !== ci) || (up >= 0 && up !== ci)) {
              const k = 1 - (1 - gridDark) * (gridW[bi] / 255);
              const v = u32[bi];
              u32[bi] = packRgba(((v & 255) * k) | 0, (((v >>> 8) & 255) * k) | 0, (((v >>> 16) & 255) * k) | 0, 255);
            }
          }
        }
      }

      // ── Pass 2: Unbroken 1-pixel maroon ink coastlines & wood sphere outline ─
      const fCopy = new Int8Array(landFlags);
      for (let py = minPy; py <= maxPy; py++) {
        const rowOff = py * bufW;
        for (let px = minPx; px <= maxPx; px++) {
          const bi = rowOff + px;
          const fl = fCopy[bi];
          const up = py > 0 ? fCopy[bi - bufW] : -1;
          const dn = py < bufH - 1 ? fCopy[bi + bufW] : -1;
          const lf = px > 0 ? fCopy[bi - 1] : -1;
          const rt = px < bufW - 1 ? fCopy[bi + 1] : -1;

          if (fl === -1) {
            if (up >= 0 || dn >= 0 || lf >= 0 || rt >= 0) u32[bi] = INK_RIM_U32;
          } else if (fl > 0) {
            if (up === 0 || dn === 0 || lf === 0 || rt === 0) {
              u32[bi] = INK_SHORE_U32;
            } else if (fl === 2 && (up === 1 || dn === 1 || lf === 1 || rt === 1)) {
              if (((px + py) & 1) === 0) u32[bi] = INK_BORDER_U32;
            }
          }
        }
      }

      bufCtx.putImageData(imgData, 0, 0);

      // ── Composite onto main display canvas ────────────────────────────────
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);

      // Celestial parchment chart rings & stars behind globe
      if (p.showScenery) {
        ctx.strokeStyle = "rgba(122,47,31,0.11)";
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 6]);
        ctx.beginPath();
        ctx.arc(cx, cy, R * 1.18, 0, Math.PI * 2);
        ctx.stroke();
        if (cam.zoom < 1.9) {
          ctx.beginPath();
          ctx.arc(cx, cy, R * 1.38, 0, Math.PI * 2);
          ctx.stroke();
        }
        ctx.setLineDash([]);

        const twT = now * 0.002;
        for (const st of stars) {
          const sx = st.x * w, sy = st.y * h;
          if (Math.hypot(sx - cx, sy - cy) < R + 18) continue;
          const alpha = 0.22 + 0.35 * (0.5 + 0.5 * Math.sin(twT + st.phase));
          ctx.fillStyle = st.cross ? `rgba(178,106,5,${alpha})` : `rgba(122,47,31,${alpha * 0.85})`;
          if (st.cross) {
            ctx.fillRect(Math.round(sx) - 3, Math.round(sy), 7, 1.5);
            ctx.fillRect(Math.round(sx), Math.round(sy) - 3, 1.5, 7);
          } else {
            ctx.fillRect(Math.round(sx), Math.round(sy), st.s, st.s);
          }
        }
      }

      // Soft warm paper drop shadow beneath the globe
      if (cam.zoom < 2.2) {
        const shadAlpha = Math.max(0, 0.12 * (1 - (cam.zoom - 1.2) / 1.0));
        ctx.fillStyle = `rgba(51,32,15,${shadAlpha})`;
        ctx.beginPath();
        ctx.ellipse(cx, cy + R + 12, R * 0.72, Math.max(8, R * 0.08), 0, 0, Math.PI * 2);
        ctx.fill();
      }

      // Vintage tabletop globe tilted meridian gimbal ring when whole globe is in frame
      if (p.showScenery && cam.zoom < 1.38) {
        const ringAlpha = Math.min(0.62, (1.38 - cam.zoom) * 1.45);
        const tilt = -0.32;
        const arcR = R + 10;
        ctx.save();
        ctx.translate(cx, cy);
        ctx.rotate(tilt);
        ctx.strokeStyle = `rgba(145,82,44,${ringAlpha})`;
        ctx.lineWidth = 4;
        ctx.beginPath();
        ctx.arc(0, 0, arcR, -Math.PI * 0.55, Math.PI * 0.55);
        ctx.stroke();
        ctx.strokeStyle = `rgba(227,178,60,${ringAlpha * 0.8})`;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(0, 0, arcR, -Math.PI * 0.55, Math.PI * 0.55);
        ctx.stroke();
        // North & South axial brass nubs
        ctx.fillStyle = `rgba(122,47,31,${ringAlpha})`;
        ctx.fillRect(-3, -arcR - 2, 6, 8);
        ctx.fillRect(-3, arcR - 6, 6, 8);
        ctx.restore();
      }

      // Blit chunky pixelated globe buffer crisp (zero blur!)
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(bufCanvas, 0, 0, bufW, bufH, 0, 0, bufW * PIXEL_STEP, bufH * PIXEL_STEP);

      // ── Pass 3: Clipped surface scenery, sprites, winds, shockwaves & Radio-Garden beacons ──
      ctx.save();
      ctx.beginPath();
      ctx.arc(cx, cy, Math.max(1, R - 2), 0, Math.PI * 2);
      ctx.clip();

      // Biome sprites across visible India cells + curving Himalayan arc
      if (p.showScenery && !story && indiaPxPerDeg >= 7.5) {
        const spriteUnit = Math.max(1, indiaPxPerDeg / 9.2);
        for (let r = 0; r < NR; r++) {
          for (let c = 0; c < NC; c++) {
            const ci = r * NC + c;
            const t = TERRAIN[ci];
            const mask = GRID.mask[r][c];
            const lat = GRID.lat[r], lon = GRID.lon[c];
            const hval = cellHash(r, c);
            let name: keyof typeof SPRITES_LIST | null = null;
            if ((t.biome === "forest" || t.biome === "highland") && hval % 4 === 0 && (mask === 2 || lon > 88)) {
              name = t.biome === "forest" ? "tree" : "pine";
            } else if (t.biome === "mountain" && (r + c) % 3 === 0) {
              const ridgeMaxLat = 35.8 - 2.2 * Math.sin(((lon - 74) / 22) * Math.PI);
              if (mask === 2 || (lon >= 72.5 && lon <= 97.5 && lat <= ridgeMaxLat)) {
                name = "peak";
              }
            } else if (t.biome === "sand" && hval % 13 === 0) {
              name = "cactus";
            }
            if (!name) continue;
            const proj = projectGeo(lat, lon, axes, R, cx, cy, 0.22);
            if (!proj.visible) continue;
            const localUnit = Math.max(1, spriteUnit * (proj.localScale / GLOBE_FOCUS_SCALE));
            stamp(ctx, name, proj.x, proj.y, localUnit * (name === "peak" ? 1.05 : 0.95));
          }
        }
      }

      // Ocean ships, sea serpent, compass rose & ocean names
      if (p.showScenery && indiaPxPerDeg >= 6.8) {
        const bob = Math.sin(tick * 0.035) * Math.max(1.2, indiaPxPerDeg * 0.12);
        const shipScale = Math.max(1, indiaPxPerDeg / 6.8);

        const arabShip = projectGeo(16.8, 68.0, axes, R, cx, cy, 0.2);
        if (arabShip.visible && !story) stamp(ctx, "ship", arabShip.x, arabShip.y + bob, shipScale);

        const bobShip = projectGeo(15.6, 87.2, axes, R, cx, cy, 0.2);
        if (bobShip.visible && !story) stamp(ctx, "ship", bobShip.x, bobShip.y - bob, shipScale * 0.88);

        const fontPx = Math.max(7, Math.min(11, Math.round(indiaPxPerDeg * 0.42)));
        ctx.fillStyle = "#3f6e78";
        ctx.font = `${fontPx}px Silkscreen, monospace`;
        ctx.textAlign = "center";

        const lblArab = projectGeo(13.4, 67.8, axes, R, cx, cy, 0.26);
        if (lblArab.visible) {
          ctx.fillText("ARABIAN", lblArab.x, lblArab.y);
          ctx.fillText("SEA", lblArab.x, lblArab.y + fontPx + 1);
        }
        const lblBob = projectGeo(12.2, 86.2, axes, R, cx, cy, 0.26);
        if (lblBob.visible) ctx.fillText("BAY OF BENGAL", lblBob.x, lblBob.y);

        const lblIo = projectGeo(2.5, 79.5, axes, R, cx, cy, 0.24);
        if (lblIo.visible) ctx.fillText("INDIAN OCEAN", lblIo.x, lblIo.y);

        const comp = projectGeo(9.8, 70.0, axes, R, cx, cy, 0.3);
        if (comp.visible && indiaPxPerDeg >= 8.2) {
          drawCompassRose(ctx, comp.x, comp.y, Math.max(8, indiaPxPerDeg * 1.05));
        }

        const serp = projectGeo(7.5, 82.8, axes, R, cx, cy, 0.28);
        if (serp.visible && indiaPxPerDeg >= 8.5) {
          const s = Math.max(2.5, indiaPxPerDeg * 0.42);
          ctx.strokeStyle = "rgba(122,47,31,0.85)";
          ctx.lineWidth = Math.max(1.3, s * 0.26);
          for (let k = 0; k < 3; k++) {
            ctx.beginPath();
            ctx.arc(serp.x + k * s * 1.1, serp.y + bob * 0.5, s * 0.52, Math.PI, 0);
            ctx.stroke();
          }
          ctx.fillStyle = "#7a2f1f";
          ctx.fillRect(serp.x + 3 * s * 1.1 - s * 0.18, serp.y + bob * 0.5 - s * 0.72, s * 0.58, s * 0.58);
          ctx.fillStyle = "#fbf5e4";
          ctx.fillRect(serp.x + 3 * s * 1.1 + s * 0.1, serp.y + bob * 0.5 - s * 0.58, s * 0.16, s * 0.16);
        }
      }

      // Wind particles curving over the 3D sphere
      if (p.showWind && p.timeline && !reducedMotion.matches) {
        ctx.lineWidth = Math.max(1, indiaPxPerDeg * 0.085);
        const uf = frame(p.timeline[p.view === "prime" ? "prime" : "branched"].u10, p.day);
        const vf = frame(p.timeline[p.view === "prime" ? "prime" : "branched"].v10, p.day);
        for (const pt of particles) {
          const r0 = Math.max(0, Math.min(NR - 1, Math.floor(pt.r)));
          const c0 = Math.max(0, Math.min(NC - 1, Math.floor(pt.c)));
          const ci = r0 * NC + c0;
          const u = uf[ci], v = vf[ci];
          const speed = Math.hypot(u, v);
          const dc = (u / 111) * 0.55, dr = (v / 111) * 0.55;
          const p0 = projectGeo(6.5 + pt.r, 65.5 + pt.c, axes, R, cx, cy, 0.08);
          pt.c += dc;
          pt.r += dr;
          pt.life += 0.012;
          const p1 = projectGeo(6.5 + pt.r, 65.5 + pt.c, axes, R, cx, cy, 0.08);
          if (p0.visible && p1.visible) {
            const a = Math.min(0.55, speed / 20) * Math.sin(Math.min(1, pt.life) * Math.PI);
            ctx.strokeStyle = `rgba(122,47,31,${a})`;
            ctx.beginPath();
            ctx.moveTo(p0.x, p0.y);
            ctx.lineTo(p1.x, p1.y);
            ctx.stroke();
          }
          if (pt.life > 1 || pt.r < 0 || pt.r >= NR || pt.c < 0 || pt.c >= NC) {
            pt.r = Math.random() * NR;
            pt.c = Math.random() * NC;
            pt.life = 0;
          }
        }
      }

      // Paint brush preview projected on India cells
      const hv = hoverRef.current;
      if (p.paintMode && hv !== null) {
        const hr = rowOf(hv), hc = colOf(hv);
        const br = p.brushRadius;
        ctx.strokeStyle = "rgba(197,61,46,0.85)";
        ctx.lineWidth = 1.4;
        for (let r = Math.floor(hr - br); r <= Math.ceil(hr + br); r++) {
          for (let c = Math.floor(hc - br); c <= Math.ceil(hc + br); c++) {
            if (!inBounds(r, c)) continue;
            const d = Math.hypot(r - hr, c - hc);
            if (d > br) continue;
            const proj = projectGeo(GRID.lat[r], GRID.lon[c], axes, R, cx, cy, 0.08);
            if (!proj.visible) continue;
            const degPx = R * DEG * proj.localScale;
            const cw = Math.max(4, degPx * Math.cos(GRID.lat[r] * DEG));
            const ch = Math.max(4, degPx);
            ctx.globalAlpha = 0.25 + 0.55 * (1 - d / Math.max(0.5, br));
            ctx.strokeRect(proj.x - cw / 2, proj.y - ch / 2, cw, ch);
          }
        }
        ctx.globalAlpha = 1;
      }

      // Expanding pixel sonar ripples & click burst on the curved globe
      for (const ring of ringRef.current) {
        drawPixelSonarRipple(
          ctx,
          ring.lat,
          ring.p,
          ring.kind,
          ring.k,
          Math.max(3, Math.round(indiaPxPerDeg * 0.34)),
          (dLat, dLon) => projectGeo(ring.lat + dLat, normalizeLon(ring.lon + dLon), axes, R, cx, cy, 0.04),
        );
      }

      // City houses & Radio-Garden glowing station beacons with de-colliding labels!
      let closestReticleCity = -1;
      let closestReticleDist = 18;
      if (p.showCities) {
        const tunedCityNow = tunedIdxRef.current;
        const labelCandidates: {
          ci: number; prio: number; x: number; y: number; dotR: number; isTuned: boolean; name: string;
        }[] = [];

        for (let ci = 0; ci < GRID.cities.length; ci++) {
          const city = GRID.cities[ci];
          const proj = projectGeo(city.lat, city.lon, axes, R, cx, cy, 0.14);
          if (!proj.visible) continue;

          const localDegPx = R * DEG * proj.localScale;
          const distReticle = Math.hypot(proj.x - cx, proj.y - cy);
          if (distReticle < closestReticleDist) {
            closestReticleDist = distReticle;
            closestReticleCity = ci;
          }

          if (p.showScenery && localDegPx >= 12.2) {
            stamp(ctx, "house", proj.x, proj.y - Math.max(4, localDegPx * 0.45), Math.max(1, localDegPx / 11));
            if (city.i % 3 === 0) {
              const puff = (tick * 0.012 + city.c / NC) % 1;
              ctx.fillStyle = `rgba(255,248,231,${(1 - puff) * 0.65})`;
              ctx.fillRect(proj.x + 2, proj.y - localDegPx * (0.7 + puff * 0.65), 2 + puff * 2, 2 + puff * 2);
            }
          }

          const isTuned = ci === tunedCityNow || p.selected === city.i;
          const pulse = 0.65 + 0.35 * Math.sin(tick * 0.05 + ci * 0.6);
          const dotR = Math.max(3, Math.min(5.5, localDegPx * 0.25)) + (isTuned ? 1.5 : 0);

          ctx.strokeStyle = isTuned
            ? `rgba(178,106,5,${0.55 * pulse})`
            : `rgba(67,130,56,${0.45 * pulse})`;
          ctx.lineWidth = isTuned ? 2 : 1.2;
          ctx.beginPath();
          ctx.arc(proj.x, proj.y, dotR + 3 + (isTuned ? 3 * pulse : 0), 0, Math.PI * 2);
          ctx.stroke();

          ctx.fillStyle = "#201509";
          ctx.fillRect(Math.round(proj.x - dotR - 1), Math.round(proj.y - dotR - 1), dotR * 2 + 2, dotR * 2 + 2);

          ctx.fillStyle = isTuned ? "#f4bd35" : "#88db52";
          ctx.fillRect(Math.round(proj.x - dotR), Math.round(proj.y - dotR), dotR * 2, dotR * 2);

          ctx.fillStyle = "#fffdf3";
          ctx.fillRect(Math.round(proj.x - dotR * 0.5), Math.round(proj.y - dotR * 0.5), 2, 2);

          if (proj.cosC > 0.32 && (isTuned || ci < 6 || localDegPx >= 10.2)) {
            const prio = isTuned ? 0 : ci < 4 ? 1 : ci < 8 ? 2 : 3;
            labelCandidates.push({
              ci, prio, x: proj.x, y: proj.y, dotR, isTuned, name: city.name.toUpperCase(),
            });
          }
        }

        // Priority-sorted 2D bounding-box de-collision for city labels
        labelCandidates.sort((a, b) => a.prio - b.prio);
        const placed: { x0: number; y0: number; x1: number; y1: number }[] = [];
        for (const item of labelCandidates) {
          ctx.font = `${item.isTuned ? 9 : 8}px Silkscreen, monospace`;
          const tw = ctx.measureText(item.name).width;
          const trySpots = [
            { lx: Math.round(item.x + item.dotR + 4), ly: Math.round(item.y + 3) },
            { lx: Math.round(item.x - item.dotR - 4 - tw), ly: Math.round(item.y + 3) },
            { lx: Math.round(item.x - tw / 2), ly: Math.round(item.y - item.dotR - 5) },
          ];
          for (const spot of trySpots) {
            const box = { x0: spot.lx - 3, y0: spot.ly - 9, x1: spot.lx + tw + 3, y1: spot.ly + 2 };
            const collides = placed.some(
              (b) => !(box.x1 < b.x0 || box.x0 > b.x1 || box.y1 < b.y0 || box.y0 > b.y1),
            );
            if (!collides) {
              placed.push(box);
              ctx.fillStyle = item.isTuned ? "rgba(248,240,218,0.94)" : "rgba(248,240,218,0.78)";
              ctx.fillRect(box.x0 + 1, box.y0 + 1, tw + 4, 10);
              ctx.fillStyle = item.isTuned ? "#7a2f1f" : "#33200f";
              ctx.textAlign = "left";
              ctx.fillText(item.name, spot.lx, spot.ly);
              break;
            }
          }
        }
      }

      // Snap tuned station automatically while spinning across cities
      if (closestReticleCity >= 0 && closestReticleCity !== tunedIdxRef.current && dragRef.current.active) {
        setTunedIdx(closestReticleCity);
      }

      // Hover cell outline
      if (hv !== null && !p.paintMode) {
        const hr = rowOf(hv), hc = colOf(hv);
        const proj = projectGeo(GRID.lat[hr], GRID.lon[hc], axes, R, cx, cy, 0.05);
        if (proj.visible) {
          const degPx = R * DEG * proj.localScale;
          const cw = Math.max(6, degPx * Math.cos(GRID.lat[hr] * DEG));
          const ch = Math.max(6, degPx);
          ctx.strokeStyle = "rgba(122,47,31,0.95)";
          ctx.lineWidth = 2;
          ctx.strokeRect(proj.x - cw / 2, proj.y - ch / 2, cw, ch);
        }
      }

      // Selected cell pulsing bracket
      if (p.selected !== null) {
        const sr = rowOf(p.selected), sc = colOf(p.selected);
        const proj = projectGeo(GRID.lat[sr], GRID.lon[sc], axes, R, cx, cy, 0.05);
        if (proj.visible) {
          const degPx = R * DEG * proj.localScale;
          const cw = Math.max(8, degPx * Math.cos(GRID.lat[sr] * DEG));
          const ch = Math.max(8, degPx);
          const pulse = 0.65 + 0.35 * selPulse.current.v;
          const x = proj.x - cw / 2, y = proj.y - ch / 2;
          const L = Math.max(4, Math.min(cw, ch) * 0.42);
          ctx.strokeStyle = `rgba(178,106,5,${pulse})`;
          ctx.lineWidth = 2;
          const corners: [number, number, number, number][] = [
            [x - 2, y - 2, 1, 1], [x + cw + 2, y - 2, -1, 1],
            [x - 2, y + ch + 2, 1, -1], [x + cw + 2, y + ch + 2, -1, -1],
          ];
          for (const [bx, by, sx, sy] of corners) {
            ctx.beginPath();
            ctx.moveTo(bx + sx * L, by);
            ctx.lineTo(bx, by);
            ctx.lineTo(bx, by + sy * L);
            ctx.stroke();
          }
        }
      }

      ctx.restore(); // end globe surface clip

      // ── Pass 4: Unclipped storybook sky scenery (clouds, birds, pixel sun) & Radio Garden reticle ─
      if (p.showScenery) {
        const sceneryAlpha = p.view === "diff" || p.paintMode ? 0.38 : 0.92;
        ctx.globalAlpha = sceneryAlpha;
        drawPixelSun(ctx, Math.min(w - 36, cx + R * 0.95), Math.max(54, cy - R * 0.86), tick, true);

        // 3 gentle drifting pixel clouds across India hemisphere
        for (let k = 0; k < 3; k++) {
          const drift = (tick * 0.0008 + k * 0.33) % 1;
          const clat = k === 0 ? 31 : k === 1 ? 19 : 25;
          const clon = 66 + drift * 32;
          const cproj = projectGeo(clat, clon, axes, R * 1.03, cx, cy, 0.25);
          if (cproj.visible) {
            const cScale = Math.max(1.2, (R * DEG * cproj.localScale) / 8.5);
            stamp(ctx, "cloud", cproj.x, cproj.y, cScale);
            if (p.lens === "flood" && k === 1) {
              ctx.fillStyle = "#2f7fb5";
              for (let drop = 0; drop < 5; drop++) {
                const fall = (tick * 0.04 + drop * 0.18) % 1;
                ctx.fillRect(cproj.x + (drop - 2) * 5, cproj.y + 10 + fall * 18, 2, 4);
              }
            }
          }
        }

        // Flapping birds over the Indian Ocean
        ctx.strokeStyle = "#69442e";
        ctx.lineWidth = 2;
        for (let k = 0; k < 3; k++) {
          const birdLon = 68 + ((tick * 0.002 + k * 0.05) % 1) * 28;
          const birdProj = projectGeo(9.5 + k * 0.9, birdLon, axes, R * 1.02, cx, cy, 0.3);
          if (birdProj.visible) {
            const flap = Math.sin(tick * 0.12 + k) * 3;
            ctx.beginPath();
            ctx.moveTo(birdProj.x - 4, birdProj.y - flap);
            ctx.lineTo(birdProj.x, birdProj.y);
            ctx.lineTo(birdProj.x + 4, birdProj.y - flap);
            ctx.stroke();
          }
        }
        ctx.globalAlpha = 1;
      }

      // Radio Garden center tuning reticle (subtle when idle, glowing green when locked near a city)
      if (!p.paintMode) {
        const locked = closestReticleCity >= 0;
        const retAlpha = locked ? 0.85 : 0.32;
        ctx.strokeStyle = locked ? `rgba(67,130,56,${retAlpha})` : `rgba(122,47,31,${retAlpha})`;
        ctx.lineWidth = locked ? 1.8 : 1.2;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.arc(cx, cy, 18, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);
        // Cross ticks
        const tickGap = 21, tickLen = 5;
        ctx.beginPath();
        ctx.moveTo(cx - tickGap, cy); ctx.lineTo(cx - tickGap + tickLen, cy);
        ctx.moveTo(cx + tickGap - tickLen, cy); ctx.lineTo(cx + tickGap, cy);
        ctx.moveTo(cx, cy - tickGap); ctx.lineTo(cx, cy - tickGap + tickLen);
        ctx.moveTo(cx, cy + tickGap - tickLen); ctx.lineTo(cx, cy + tickGap);
        ctx.stroke();
      }

      // Running simulation pulse bar at bottom of globe
      if (p.running) {
        const t = (tick % 60) / 60;
        const barW = 140;
        const segW = 36;
        ctx.fillStyle = "rgba(29,111,163,0.78)";
        ctx.fillRect(cx - barW / 2 + t * (barW - segW), Math.min(h - 14, cy + R + 12), segW, 2.5);
      }

      raf = requestAnimationFrame(draw);
    };

    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [size, globeGeom, showWind]);

  // Sprite name map reference for type narrowing above
  const SPRITES_LIST = { tree: 1, pine: 1, peak: 1, cactus: 1 } as const;

  /* ── pointer hit-testing against curved India grid & city beacons ── */
  const hitTestPoint = useCallback((clientX: number, clientY: number) => {
    const canvas = canvasRef.current;
    if (!canvas || size.w === 0) return { cell: null as number | null, cityIndex: -1, onSphere: false, geo: null as { lat: number; lon: number } | null, mx: 0, my: 0 };
    const rect = canvas.getBoundingClientRect();
    const mx = clientX - rect.left;
    const my = clientY - rect.top;
    const cam = camRef.current;
    const { R, cx, cy } = globeGeom(size.w, size.h, cam.zoom);
    const axes = cameraAxes(cam.lat, cam.lon);

    // Check city beacons first (generous click radius in non-paint mode!)
    let cityIndex = -1;
    if (showCities && !paintMode) {
      let bestD = 16;
      for (let ci = 0; ci < GRID.cities.length; ci++) {
        const city = GRID.cities[ci];
        const proj = projectGeo(city.lat, city.lon, axes, R, cx, cy, 0.14);
        if (!proj.visible) continue;
        const d = Math.hypot(mx - proj.x, my - proj.y);
        if (d < bestD) { bestD = d; cityIndex = ci; }
      }
    }

    const geo = unprojectGeo(mx, my, axes, R, cx, cy);
    if (!geo) return { cell: null, cityIndex, onSphere: false, geo: null as { lat: number; lon: number } | null, mx, my };

    let cell: number | null = null;
    if (geo.lat >= 6.0 && geo.lat < 38.0 && geo.lon >= 65.0 && geo.lon < 101.0) {
      const r = Math.floor(geo.lat - 6.0);
      const c = Math.floor(geo.lon - 65.0);
      if (inBounds(r, c)) cell = idx(r, c);
    }
    return { cell, cityIndex, onSphere: true, geo: { lat: geo.lat, lon: geo.lon }, mx, my };
  }, [size, globeGeom, showCities, paintMode]);

  const handleDown = (e: React.PointerEvent) => {
    flightAnimRef.current?.pause();
    const hit = hitTestPoint(e.clientX, e.clientY);
    (e.target as HTMLElement).setPointerCapture?.(e.pointerId);

    const wantsSpinInPaint = e.button === 2 || e.button === 1 || hit.cell === null;
    if (paintMode && !wantsSpinInPaint && hit.cell !== null) {
      const erase = e.shiftKey;
      dragRef.current = {
        ...dragRef.current,
        active: true,
        mode: "paint",
        startX: e.clientX,
        startY: e.clientY,
        lastX: e.clientX,
        lastY: e.clientY,
        movedDist: 0,
        erase,
        lastCell: hit.cell,
      };
      onPaint(brushCellsAround(hit.cell, brushRadius), erase);
      return;
    }

    setAutoSpin(false);
    velRef.current = { lat: 0, lon: 0 };
    dragRef.current = {
      active: true,
      mode: "spin",
      startX: e.clientX,
      startY: e.clientY,
      lastX: e.clientX,
      lastY: e.clientY,
      startLat: camRef.current.lat,
      startLon: camRef.current.lon,
      movedDist: 0,
      erase: false,
      lastCell: null,
    };
  };

  const handleMove = (e: React.PointerEvent) => {
    const hit = hitTestPoint(e.clientX, e.clientY);
    setHover(hit.cell);
    setCursorPx(hit.cell !== null ? { x: hit.mx, y: hit.my } : null);

    const dr = dragRef.current;
    if (!dr.active) return;

    if (dr.mode === "paint") {
      if (hit.cell === null) return;
      const last = dr.lastCell;
      if (last !== null && last !== hit.cell) {
        const r0 = rowOf(last), c0 = colOf(last), r1 = rowOf(hit.cell), c1 = colOf(hit.cell);
        const steps = Math.max(Math.abs(r1 - r0), Math.abs(c1 - c0));
        for (let s = 1; s <= steps; s++) {
          const rr = Math.round(r0 + ((r1 - r0) * s) / steps);
          const cc = Math.round(c0 + ((c1 - c0) * s) / steps);
          onPaint(brushCellsAround(idx(rr, cc), brushRadius), dr.erase);
        }
      } else {
        onPaint(brushCellsAround(hit.cell, brushRadius), dr.erase);
      }
      dr.lastCell = hit.cell;
      return;
    }

    // Spherical drag rotation scaled cleanly by zoom & sphere radius
    const { R } = globeGeom(size.w, size.h, camRef.current.zoom);
    const degPerPx = 115 / Math.max(120, R);
    const totalDx = e.clientX - dr.startX;
    const totalDy = e.clientY - dr.startY;
    const stepDx = e.clientX - dr.lastX;
    const stepDy = e.clientY - dr.lastY;
    dr.lastX = e.clientX;
    dr.lastY = e.clientY;
    dr.movedDist += Math.hypot(stepDx, stepDy);

    camRef.current.lon = normalizeLon(dr.startLon - totalDx * degPerPx);
    camRef.current.lat = Math.max(-75, Math.min(75, dr.startLat + totalDy * degPerPx));
    velRef.current = {
      lon: Math.max(-4, Math.min(4, -stepDx * degPerPx * 0.72)),
      lat: Math.max(-4, Math.min(4, stepDy * degPerPx * 0.72)),
    };
  };

  const handleUp = (e: React.PointerEvent) => {
    const dr = dragRef.current;
    if (!dr.active) return;
    const wasPaint = dr.mode === "paint";
    const wasClick = dr.movedDist < 6;
    dr.active = false;
    dr.lastCell = null;

    if (wasPaint) {
      onPaintEnd();
      return;
    }

    if (wasClick) {
      velRef.current = { lat: 0, lon: 0 };
      const hit = hitTestPoint(e.clientX, e.clientY);
      if (hit.cityIndex >= 0) {
        const cty = GRID.cities[hit.cityIndex];
        if (cty.i === selected) spawnGeoRipple(cty.lat, cty.lon, "select");
        tuneCity(hit.cityIndex, true);
      } else if (hit.cell !== null) {
        if (hit.cell === selected) {
          spawnGeoRipple(GRID.lat[rowOf(hit.cell)], GRID.lon[colOf(hit.cell)], "select");
          onSelect(null);
        } else {
          onSelect(hit.cell);
        }
      } else if (hit.onSphere && hit.geo) {
        spawnGeoRipple(hit.geo.lat, hit.geo.lon, "select");
      }
    }
  };

  const handleWheel = (e: React.WheelEvent) => {
    flightAnimRef.current?.pause();
    const factor = e.deltaY < 0 ? 1.14 : 0.88;
    camRef.current.zoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, camRef.current.zoom * factor));
  };

  const handleDoubleClick = (e: React.PointerEvent) => {
    if (paintMode) return;
    const hit = hitTestPoint(e.clientX, e.clientY);
    if (hit.cityIndex >= 0) {
      tuneCity(hit.cityIndex, true);
      return;
    }
    const cam = camRef.current;
    const { R, cx, cy } = globeGeom(size.w, size.h, cam.zoom);
    const axes = cameraAxes(cam.lat, cam.lon);
    const canvas = canvasRef.current;
    if (canvas) {
      const rect = canvas.getBoundingClientRect();
      const geo = unprojectGeo(e.clientX - rect.left, e.clientY - rect.top, axes, R, cx, cy);
      if (geo) {
        flyTo(geo.lat, geo.lon, Math.min(MAX_ZOOM, cam.zoom + 0.5), 520);
        return;
      }
    }
    flyTo(INDIA_HOME.lat, INDIA_HOME.lon, Math.min(MAX_ZOOM, cam.zoom + 0.5), 520);
  };

  /* ── Hover info tooltip data ── */
  const hoverInfo = useMemo(() => {
    if (hover === null || !timeline) return null;
    const r = rowOf(hover), c = colOf(hover);
    const lsm = landSeaMask?.[r]?.[c] ?? GRID.land_sea_mask[r][c];
    const raw = lens && lensOut
      ? (view === "prime" ? lensOut.base[hover] : view === "branched" ? lensOut.value[hover] : lensOut.delta[hover])
      : view === "diff"
        ? diffFrame(timeline, variable, day)[hover]
        : frame(timeline[view][variable], day)[hover];
    const unit = lens ? (lens === "heat" ? "°C Tw" : lens === "flood" ? "FRI" : "mm/day") : displayUnit(variable);
    const shown = lens ? raw : view === "diff" ? displayDelta(variable, raw) : display(variable, raw);
    const nc = nearestCity(r, c);
    const rawDelta = !lens && view !== "diff"
      ? frame(timeline.branched[variable], day)[hover] - frame(timeline.prime[variable], day)[hover]
      : null;
    return {
      r, c, lat: GRID.lat[r], lon: GRID.lon[c], region: regionOf(r, c, lsm),
      value: shown, unit, mask: GRID.mask[r][c],
      city: nc && nc.distanceCells <= 1.5 ? nc.city : null,
      delta: rawDelta !== null ? displayDelta(variable, rawDelta) : null,
    };
  }, [hover, timeline, landSeaMask, view, variable, day, lens, lensOut]);

  /* ── Live tuned station readout (Radio Garden card!) ── */
  const stationInfo = useMemo(() => {
    const city = GRID.cities[tunedIdx] ?? GRID.cities[0];
    if (!timeline) return { city, valStr: "—", unit: "", deltaStr: null, windStr: "—", tC: null, tp: null };
    const i = city.i;
    const src = view === "prime" ? "prime" : "branched";
    const tK = frame(timeline[src].t2m, day)[i];
    const tp = frame(timeline[src].tp, day)[i];
    const u = frame(timeline[src].u10, day)[i];
    const v = frame(timeline[src].v10, day)[i];
    const wm = windMetrics(u, v);

    const rawVal = lens && lensOut
      ? (view === "prime" ? lensOut.base[i] : view === "branched" ? lensOut.value[i] : lensOut.delta[i])
      : view === "diff"
        ? diffFrame(timeline, variable, day)[i]
        : frame(timeline[view][variable], day)[i];
    const unit = lens ? (lens === "heat" ? "°C Tw" : lens === "flood" ? "FRI" : "mm/d") : displayUnit(variable);
    const shown = lens ? rawVal : view === "diff" ? displayDelta(variable, rawVal) : display(variable, rawVal);
    const rawDelta = !lens && view !== "diff"
      ? frame(timeline.branched[variable], day)[i] - frame(timeline.prime[variable], day)[i]
      : null;
    const deltaVal = rawDelta !== null ? displayDelta(variable, rawDelta) : null;

    return {
      city,
      valStr: shown.toFixed(Math.abs(shown) < 10 ? 2 : 1),
      unit,
      deltaStr: deltaVal !== null && Math.abs(deltaVal) > 1e-3 ? `${deltaVal > 0 ? "+" : ""}${deltaVal.toFixed(2)}` : null,
      windStr: `${wm.fromCompass} ${wm.speed.toFixed(1)} m/s`,
      tC: (tK - 273.15).toFixed(1),
      tp: tp.toFixed(1),
    };
  }, [tunedIdx, timeline, view, variable, day, lens, lensOut]);

  return (
    <div
      ref={wrapRef}
      className="relative h-full w-full scanlines select-none"
      style={{
        background: "radial-gradient(circle at 50% 48%, #f8f1dc 0%, #f3e9cf 56%, #e7d8b0 100%)",
      }}
    >
      <canvas
        ref={canvasRef}
        onPointerDown={handleDown}
        onPointerMove={handleMove}
        onPointerUp={handleUp}
        onPointerCancel={() => { dragRef.current.active = false; }}
        onLostPointerCapture={() => {
          if (dragRef.current.active && dragRef.current.mode === "paint") onPaintEnd();
          dragRef.current.active = false;
        }}
        onPointerLeave={() => {
          setHover(null);
          setCursorPx(null);
        }}
        onWheel={handleWheel}
        onDoubleClick={handleDoubleClick as unknown as React.MouseEventHandler}
        onContextMenu={(e) => e.preventDefault()}
        className={`block h-full w-full touch-none ${
          paintMode ? "cursor-crosshair" : "cursor-grab active:cursor-grabbing"
        }`}
      />

      <canvas ref={celebrationRef} aria-hidden="true" className="pointer-events-none absolute inset-0 z-20 h-full w-full" />

      <AnimatePresence>
        {celebrating && (
          <motion.div
            initial={{ opacity: 0, y: 20, scale: 0.8 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -20 }}
            className="pointer-events-none absolute left-1/2 top-14 z-30 -translate-x-1/2 border-2 border-inkline bg-panel px-5 py-3 font-pixel text-xs text-shock shadow-[4px_4px_0_#69442e]"
          >
            A NEW TIMELINE!
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── right-edge globe navigation dock (zoom + center India + spin) ── */}
      <div className="pointer-events-auto absolute right-2 top-24 z-20 flex flex-col gap-1 sm:top-14">
        <button
          onClick={() => flyTo(camRef.current.lat, camRef.current.lon, Math.min(MAX_ZOOM, camRef.current.zoom + 0.55), 320)}
          className="grid size-7 place-items-center border border-grid-line bg-panel/95 text-ink-dim shadow-[2px_2px_0_rgba(51,32,15,0.14)] hover:border-prime hover:text-prime"
          title="Zoom in globe"
          aria-label="Zoom in"
        >
          <Plus size={13} />
        </button>
        <button
          onClick={() => flyTo(camRef.current.lat, camRef.current.lon, Math.max(MIN_ZOOM, camRef.current.zoom - 0.55), 320)}
          className="grid size-7 place-items-center border border-grid-line bg-panel/95 text-ink-dim shadow-[2px_2px_0_rgba(51,32,15,0.14)] hover:border-prime hover:text-prime"
          title="Zoom out globe"
          aria-label="Zoom out"
        >
          <Minus size={13} />
        </button>
        <button
          onClick={() => { setAutoSpin(false); flyTo(INDIA_HOME.lat, INDIA_HOME.lon, INDIA_HOME.zoom, 650); }}
          className="grid size-7 place-items-center border border-grid-line bg-panel/95 text-inkline shadow-[2px_2px_0_rgba(51,32,15,0.14)] hover:border-inkline hover:bg-abyss"
          title="Center India on globe"
          aria-label="Center India"
        >
          <Compass size={13} />
        </button>
        <button
          onClick={() => setAutoSpin((s) => !s)}
          className={`grid size-7 place-items-center border shadow-[2px_2px_0_rgba(51,32,15,0.14)] transition-colors ${
            autoSpin
              ? "border-safe bg-safe/20 text-safe"
              : "border-grid-line bg-panel/95 text-ink-dim hover:text-ink"
          }`}
          title="Toggle globe auto-orbit"
          aria-label="Toggle auto spin"
        >
          <RotateCw size={12} className={autoSpin ? "animate-spin" : ""} style={{ animationDuration: "5s" }} />
        </button>
        <button
          onClick={onSwitchToFlatMap}
          className="grid size-7 place-items-center border border-grid-line bg-panel/95 text-prime shadow-[2px_2px_0_rgba(51,32,15,0.14)] hover:border-prime hover:bg-prime/10"
          title="Open 2D Flat Grid Map"
          aria-label="Switch to flat grid map"
        >
          <Maximize2 size={12} />
        </button>
      </div>

      {/* ── Radio Garden live city station tuner card (bottom-right) ── */}
      {showCities && (
        <div className="pointer-events-auto absolute bottom-2 right-2 z-20 w-[212px] panel px-2.5 py-2 sm:w-[244px]">
          <div className="flex items-center justify-between gap-1.5">
            <span className="flex items-center gap-1.5 font-pixel text-[8px] text-safe">
              <span className="inline-block size-2 border border-ink bg-[#88db52] mm-pulse" />
              <Radio size={9} />
              TUNED STATION
            </span>
            <div className="flex items-center gap-0.5">
              <button
                onClick={() => tuneCity(tunedIdx - 1)}
                className="grid size-5 place-items-center border border-grid-line bg-abyss text-ink-dim hover:text-ink"
                title="Previous city station"
                aria-label="Previous station"
              >
                <ChevronLeft size={11} />
              </button>
              <button
                onClick={() => {
                  const next = (tunedIdx + 1 + Math.floor(Math.random() * (GRID.cities.length - 1))) % GRID.cities.length;
                  tuneCity(next);
                }}
                className="grid size-5 place-items-center border border-grid-line bg-abyss text-ink-dim hover:text-branch"
                title="Tune random city across India"
                aria-label="Random station"
              >
                <Shuffle size={10} />
              </button>
              <button
                onClick={() => tuneCity(tunedIdx + 1)}
                className="grid size-5 place-items-center border border-grid-line bg-abyss text-ink-dim hover:text-ink"
                title="Next city station"
                aria-label="Next station"
              >
                <ChevronRight size={11} />
              </button>
            </div>
          </div>

          <div className="mt-1 flex items-baseline justify-between gap-2">
            <button
              onClick={() => tuneCity(tunedIdx, true)}
              className="truncate text-left font-pixel text-[10px] text-ink hover:text-prime"
              title="Center & inspect station"
            >
              {stationInfo.city.name.toUpperCase()}
            </button>
            <span className="shrink-0 font-mono text-[8px] text-ink-faint">{stationInfo.city.note}</span>
          </div>

          <div className="mt-1 flex items-baseline justify-between gap-2 border-t border-grid-line pt-1">
            <div className="flex items-baseline gap-1">
              <span className="font-mono text-[15px] font-medium text-ink tabular-nums">
                {stationInfo.valStr}
              </span>
              <span className="font-mono text-[9px] text-ink-dim">{stationInfo.unit}</span>
              {stationInfo.deltaStr && (
                <span className={`ml-1 font-mono text-[9px] ${Number(stationInfo.deltaStr) > 0 ? "text-branch" : "text-prime"}`}>
                  Δ{stationInfo.deltaStr}
                </span>
              )}
            </div>
            <button
              onClick={() => onSelect(stationInfo.city.i)}
              className="flex items-center gap-0.5 font-pixel text-[7px] text-branch hover:underline"
              title="Open full diagnostics in Cell Inspector"
            >
              <Crosshair size={8} /> INSPECT
            </button>
          </div>

          {stationInfo.tC !== null && (
            <div className="mt-0.5 hidden justify-between font-mono text-[8px] text-ink-faint sm:flex">
              <span>{stationInfo.tC}°C · {stationInfo.tp} mm/d</span>
              <span>{stationInfo.windStr}</span>
            </div>
          )}
        </div>
      )}

      {/* ── Paint-on-globe subtle banner ── */}
      <AnimatePresence>
        {paintMode && (
          <motion.div
            initial={{ opacity: 0, y: -6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            className="pointer-events-none absolute left-1/2 top-12 z-20 -translate-x-1/2 border border-shock/60 bg-panel/95 px-2.5 py-1 font-pixel text-[8px] text-shock shadow-[2px_2px_0_rgba(51,32,15,0.14)]"
          >
            PAINTING ON GLOBE · DRAG INDIA TO SHOCK · SHIFT+DRAG ERASE
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Hover cell pill (matches PixelGrid) ── */}
      <AnimatePresence>
        {hoverInfo && cursorPx && (
          <motion.div
            key="globe-hover-pill"
            initial={{ opacity: 0, scale: 0.94, y: 4 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.96 }}
            transition={{ type: "spring", stiffness: 520, damping: 32 }}
            className="pointer-events-none absolute z-20 w-[196px] panel px-2.5 py-2 text-[10px] leading-tight"
            style={{
              left: cursorPx.x > size.w - 220 ? Math.max(8, cursorPx.x - 208) : cursorPx.x + 14,
              top: cursorPx.y < 110 ? cursorPx.y + 18 : cursorPx.y - 90,
            }}
          >
            <div className="flex items-baseline justify-between gap-2">
              <span className="font-pixel text-[9px] text-prime">
                {hoverInfo.city ? hoverInfo.city.name.toUpperCase() : `CELL ${hoverInfo.r},${hoverInfo.c}`}
              </span>
              <span className="text-ink-faint">{fmtLat(hoverInfo.lat)} {fmtLon(hoverInfo.lon)}</span>
            </div>
            <div className="mt-1 flex items-baseline gap-1">
              <span className="font-mono text-[17px] font-medium text-ink tabular-nums">
                {hoverInfo.value.toFixed(Math.abs(hoverInfo.value) < 10 ? 2 : 1)}
              </span>
              <span className="text-ink-dim">{hoverInfo.unit}</span>
            </div>
            {hoverInfo.delta !== null && Math.abs(hoverInfo.delta) > 1e-4 && (
              <div className={`mt-0.5 font-mono ${hoverInfo.delta > 0 ? "text-branch" : "text-prime"}`}>
                Δ {hoverInfo.delta > 0 ? "+" : ""}{hoverInfo.delta.toFixed(2)} vs prime
              </div>
            )}
            <div className="mt-1 flex items-center gap-1.5 border-t border-grid-line pt-1 text-ink-faint">
              <span
                className="inline-block size-1.5"
                style={{
                  background: hoverInfo.mask === 2 ? "var(--color-safe)" : hoverInfo.mask === 0 ? "#4a9ab8" : "var(--color-ink-faint)",
                }}
              />
              {hoverInfo.region}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {!timeline && (
        <div className="pointer-events-none absolute inset-0 grid place-items-center">
          <div className="panel px-4 py-3 text-center">
            <div className="font-pixel text-[11px] text-ink-dim mm-pulse">AWAITING /simulate</div>
            <div className="mt-1.5 font-mono text-[10px] text-ink-faint">paint a shock, then branch the timeline</div>
          </div>
        </div>
      )}
    </div>
  );
}
