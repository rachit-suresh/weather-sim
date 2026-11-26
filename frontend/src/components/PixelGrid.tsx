import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { animate, createTimer, stagger } from "animejs";
import { motion, AnimatePresence, useReducedMotion } from "motion/react";
import confetti from "canvas-confetti";
import {
  CELLS, GRID, NC, NR, colOf, rowOf, idx,
  fmtLat, fmtLon, regionOf, nearestCity, display, displayDelta, displayUnit,
  type VarKey,
} from "@/lib/grid";
import { frame, diffFrame, type Timeline } from "@/lib/api";
import { rgbCss, type RGB } from "@/lib/colormaps";
import {
  PAPER, TERRAIN, cellHash, stamp, computeTargetField,
  colorForFieldValue, brushCellsAround, drawPixelSun, drawCompassRose, drawPixelSonarRipple,
} from "@/lib/mapSprites";
import type { LensId, LensOutput } from "@/lib/impact";

export type ViewMode = "prime" | "branched" | "diff";

export interface Ripple { id: number; r: number; c: number; kind: "shock" | "run" | "select" }

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
  running: boolean;
  focusRegion: string | null;
}

const GAP = 1.2;                     // px between pixels — the pixel-art signature

export default function PixelGrid(props: Props) {
  const reduceMotion = useReducedMotion();
  const {
    timeline, landSeaMask, view, variable, day, lens, lensOut, paintMask, paintMode, brushRadius,
    onPaint, onPaintEnd, selected, onSelect, ripples, showWind, running, focusRegion,
  } = props;

  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const celebrationRef = useRef<HTMLCanvasElement>(null);
  const cannonRef = useRef<confetti.CreateTypes | null>(null);
  const [celebrating, setCelebrating] = useState(false);
  const [hover, setHover] = useState<number | null>(null);
  const [geom, setGeom] = useState({ cell: 16, ox: 0, oy: 0, w: 0, h: 0 });

  // ── animated display state ────────────────────────────────────────────────
  const fromRef = useRef(new Float32Array(CELLS));
  const toRef = useRef(new Float32Array(CELLS));
  const dispRef = useRef(new Float32Array(CELLS));
  const blendRef = useRef({ v: 1 });
  const scaleRef = useRef({ from: 1, to: 1, v: 1 });
  const hoverGlow = useRef({ v: 0 });
  const selPulse = useRef({ v: 0 });
  const ringRef = useRef<{ r: number; c: number; p: number; kind: Ripple["kind"]; k: number }[]>([]);
  const seenRipplesRef = useRef<Set<number> | null>(null);
  const activeRippleAnimsRef = useRef(new Set<ReturnType<typeof animate>>());
  const sweepRef = useRef({ v: 1 });
  const paintingRef = useRef(false);
  const eraseRef = useRef(false);
  const lastCellRef = useRef<number | null>(null);
  const propsRef = useRef(props);
  propsRef.current = props;
  const hoverRef = useRef<number | null>(null);
  hoverRef.current = hover;

  /* ── target field for the current view / variable / day / lens ───────────── */
  const target = useMemo(
    () => computeTargetField(timeline, view, variable, day, lens, lensOut),
    [timeline, view, variable, day, lens, lensOut],
  );

  /* Retarget + tween whenever the target changes. */
  useEffect(() => {
    fromRef.current.set(dispRef.current);
    toRef.current.set(target.values);
    scaleRef.current.from = scaleRef.current.v;
    scaleRef.current.to = target.scale;
    if (reduceMotion) { blendRef.current.v = 1; scaleRef.current.v = target.scale; return; }
    blendRef.current.v = 0;
    const a = animate(blendRef.current, { v: 1, duration: 420, ease: "outCubic" });
    const b = animate(scaleRef.current, { v: target.scale, duration: 420, ease: "outCubic" });
    return () => { a.pause(); b.pause(); };
  }, [target, reduceMotion]);

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
    void cannonRef.current?.({ particleCount: 48, spread: 85, startVelocity: 24, gravity: 0.8, ticks: 95, flat: true, shapes: ["square"], colors: ["#efb52f", "#ce4d34", "#438238", "#529db5"], origin: { x: 0.5, y: 0.65 } });
    const timeout = window.setTimeout(() => setCelebrating(false), 2000);
    return () => { window.clearTimeout(timeout); setCelebrating(false); };
  }, [timeline?.runId, props.showScenery, reduceMotion]);

  useEffect(() => {
    if (!props.showScenery || reduceMotion) {
      cannonRef.current?.reset();
      setCelebrating(false);
    }
  }, [props.showScenery, reduceMotion]);

  useEffect(() => {
    if (!timeline || reduceMotion) { sweepRef.current.v = 1; return; }
    sweepRef.current.v = 0;
    const a = animate(sweepRef.current, { v: 1, duration: 900, ease: "inOutQuad" });
    return () => { a.pause(); };
  }, [timeline?.runId, reduceMotion]);

  const spawnCellRipple = useCallback((r: number, c: number, kind: Ripple["kind"]) => {
    if (reduceMotion) return;
    const objs = [0, 1, 2].map((k) => ({ r, c, p: 0, kind, k }));
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

  /* Ripples: pixel sonar rings, staggered and non-cancelling. */
  useEffect(() => {
    if (seenRipplesRef.current === null) {
      seenRipplesRef.current = new Set(ripples.map((r) => r.id));
      return;
    }
    for (const r of ripples) {
      if (seenRipplesRef.current.has(r.id)) continue;
      seenRipplesRef.current.add(r.id);
      spawnCellRipple(r.r, r.c, r.kind);
    }
  }, [ripples, spawnCellRipple]);

  useEffect(() => {
    return () => {
      for (const a of activeRippleAnimsRef.current) a.pause();
      activeRippleAnimsRef.current.clear();
      ringRef.current = [];
    };
  }, []);

  /* Hover / selection glow springs. */
  useEffect(() => {
    if (reduceMotion) { hoverGlow.current.v = hover !== null ? 1 : 0; return; }
    const a = animate(hoverGlow.current, { v: hover !== null ? 1 : 0, duration: 180, ease: "outQuad" });
    return () => { a.pause(); };
  }, [hover, reduceMotion]);
  useEffect(() => {
    if (selected === null || reduceMotion) { selPulse.current.v = 0; return; }
    const t = createTimer({
      duration: 1600, loop: true,
      onUpdate: (self) => { selPulse.current.v = 0.5 + 0.5 * Math.sin((self.currentTime / 1600) * Math.PI * 2); },
    });
    return () => { t.pause(); };
  }, [selected, reduceMotion]);

  /* ── layout ─────────────────────────────────────────────────────────────── */
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      const rect = el.getBoundingClientRect();
      const pad = rect.width < 500 ? 12 : 24;
      const top = rect.width < 500 ? 65 : 76;
      const bottom = 44;
      const cell = Math.max(3, Math.min((rect.width - pad * 2) / NC, (rect.height - top - bottom) / NR));
      const w = cell * NC, h = cell * NR;
      setGeom({ cell, ox: (rect.width - w) / 2, oy: top + Math.max(0, (rect.height - top - bottom - h) / 2), w: rect.width, h: rect.height });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  /* ── data colour for one cell (terrain shows through via overlay alpha) ─── */
  const colorFor = useCallback((v: number): RGB => {
    const p = propsRef.current;
    return colorForFieldValue(v, scaleRef.current.v, p.view, p.variable, p.lens, p.lensOut);
  }, []);

  /* ── main render loop ───────────────────────────────────────────────────── */
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || geom.w === 0) return;
    const ctx = canvas.getContext("2d", { alpha: true });
    if (!ctx) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = Math.round(geom.w * dpr);
    canvas.height = Math.round(geom.h * dpr);
    canvas.style.width = `${geom.w}px`;
    canvas.style.height = `${geom.h}px`;

    let raf = 0;
    let tick = 0;
    const particles: { r: number; c: number; life: number }[] = [];
    if (showWind) {
      for (let k = 0; k < 150; k++)
        particles.push({ r: Math.random() * NR, c: Math.random() * NC, life: Math.random() });
    }

    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
    // Backend row 0 = south (6.5°N), row 31 = north (37.5°N) → render north at top
    const screenY = (r: number, cell: number, oy: number, half: number) =>
      oy + (NR - 1 - r) * cell + half;

    const draw = () => {
      tick = reducedMotion.matches ? 0 : performance.now() * 0.06;
      const p = propsRef.current;
      const { cell, ox, oy } = geom;
      const blend = blendRef.current.v;
      const from = fromRef.current, to = toRef.current, disp = dispRef.current;
      for (let i = 0; i < CELLS; i++) disp[i] = from[i] + (to[i] - from[i]) * blend;

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, geom.w, geom.h);

      const px = Math.max(1, cell - GAP);
      const half = GAP / 2;

      let focusSet: Set<number> | null = null;
      if (p.focusRegion) {
        const reg = GRID.regions.find((x) => x.name === p.focusRegion);
        if (reg) focusSet = new Set(reg.cells.map(([r, c]) => idx(r, c)));
      }

      // ── cells: terrain base + data painted on top ──
      const strong = p.view === "diff";
      for (let r = 0; r < NR; r++) {
        const y = screenY(r, cell, oy, half);
        for (let c = 0; c < NC; c++) {
          const i = r * NC + c;
          const mask = GRID.mask[r][c];
          const t = TERRAIN[i];
          let base = t.base;

          if (mask === 2) {
            const b = 1 + 0.03 * Math.sin(tick * 0.02 + (r + c) * 0.35);
            base = [base[0] * b, base[1] * b, base[2] * b];
          }

          const x = ox + c * cell + half;
          ctx.fillStyle = rgbCss(base);
          ctx.fillRect(x, y, px, px);

          if (p.timeline) {
            const col = colorFor(disp[i]);
            const a = !p.showScenery ? 1 : mask === 2 ? (strong ? 0.92 : 0.62) : mask === 0 ? (strong ? 0.65 : 0.16) : (strong ? 0.45 : 0.18);
            ctx.fillStyle = rgbCss(col, a);
            ctx.fillRect(x, y, px, px);
          }
          if (p.showScenery && !strong && mask !== 0) {
            ctx.fillStyle = "rgba(255,248,231,0.16)";
            ctx.fillRect(x, y, px, 1);
            ctx.fillStyle = "rgba(105,68,46,0.12)";
            ctx.fillRect(x, y + px - 1, px, 1);
          }

          if (focusSet && !focusSet.has(i)) {
            ctx.fillStyle = rgbCss(PAPER, 0.74);
            ctx.fillRect(x, y, px, px);
          }

          const sw = sweepRef.current.v;
          if (sw < 1) {
            const front = sw * (NC + 6) - 3;
            const d = Math.abs(c - front);
            if (d < 2.2) {
              ctx.fillStyle = `rgba(255,252,235,${0.5 * (1 - d / 2.2)})`;
              ctx.fillRect(x, y, px, px);
            }
          }
        }
      }

      if (p.showScenery && cell >= 7) {
        for (let r = 0; r < NR; r++) {
          const y = screenY(r, cell, oy, half);
          for (let c = 0; c < NC; c++) {
            const i = r * NC + c;
            const t = TERRAIN[i];
            const h = cellHash(r, c);
            const x = ox + c * cell + half;
            if (t.biome === "ocean" && h % 5 === 0) {
              ctx.fillStyle = "rgba(251,245,228,0.6)";
              const w = px * 0.34, hh = Math.max(1, px * 0.09);
              ctx.fillRect(x + px * 0.14, y + px * 0.3, w, hh);
              ctx.fillRect(x + px * 0.5, y + px * 0.62, w * 0.8, hh);
            } else if ((t.biome === "forest" || t.biome === "highland") && h % 4 === 0) {
              stamp(ctx, t.biome === "forest" ? "tree" : "pine", x + px / 2, y + px / 2, cell / 10);
            } else if (t.biome === "mountain" && (r + c) % 3 === 0) {
              stamp(ctx, "peak", x + px / 2, y + px / 2, cell / 9);
            } else if (t.biome === "sand" && h % 13 === 0) {
              stamp(ctx, "cactus", x + px / 2, y + px / 2, cell / 11);
            } else if (t.biome === "sand" && h % 4 === 0) {
              ctx.fillStyle = "rgba(192,122,8,0.4)";
              ctx.fillRect(x + px * 0.2, y + px * 0.55, px * 0.4, Math.max(1, px * 0.08));
            } else if (t.biome === "grass" && h % 7 === 0) {
              ctx.fillStyle = "rgba(63,125,44,0.55)";
              const s = Math.max(1, px * 0.1);
              ctx.fillRect(x + px * 0.3, y + px * 0.5, s, s * 1.6);
              ctx.fillRect(x + px * 0.55, y + px * 0.42, s, s * 2);
            }
          }
        }
      }

      // ── ink coastlines: maroon where land meets sea, dashed India border ──
      const isSeaCell = (rr: number, cc: number) =>
        rr < 0 || cc < 0 || rr >= NR || cc >= NC
          ? false
          : (p.landSeaMask?.[rr]?.[cc] ?? GRID.land_sea_mask[rr][cc]) < 0.45 && GRID.mask[rr][cc] === 0;

      ctx.strokeStyle = "rgba(122,47,31,0.9)";
      ctx.lineWidth = Math.max(1.5, cell * 0.1);
      ctx.beginPath();
      for (let r = 0; r < NR; r++) {
        const y = screenY(r, cell, oy, half);
        for (let c = 0; c < NC; c++) {
          if (GRID.mask[r][c] === 0) continue;
          const x = ox + c * cell + half;
          // north neighbor is r + 1 (drawn at top edge y), south neighbor is r - 1 (drawn at bottom edge y + px)
          if (isSeaCell(r + 1, c)) { ctx.moveTo(x, y); ctx.lineTo(x + px, y); }
          if (isSeaCell(r - 1, c)) { ctx.moveTo(x, y + px); ctx.lineTo(x + px, y + px); }
          if (isSeaCell(r, c - 1)) { ctx.moveTo(x, y); ctx.lineTo(x, y + px); }
          if (isSeaCell(r, c + 1)) { ctx.moveTo(x + px, y); ctx.lineTo(x + px, y + px); }
        }
      }
      ctx.stroke();

      ctx.strokeStyle = "rgba(122,47,31,0.4)";
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      for (let r = 0; r < NR; r++) {
        const y = screenY(r, cell, oy, half);
        for (let c = 0; c < NC; c++) {
          if (GRID.mask[r][c] !== 2) continue;
          const x = ox + c * cell + half;
          const foreign = (rr: number, cc: number) =>
            rr >= 0 && cc >= 0 && rr < NR && cc < NC && GRID.mask[rr][cc] === 1;
          if (foreign(r + 1, c)) { ctx.moveTo(x, y); ctx.lineTo(x + px, y); }
          if (foreign(r - 1, c)) { ctx.moveTo(x, y + px); ctx.lineTo(x + px, y + px); }
          if (foreign(r, c - 1)) { ctx.moveTo(x, y); ctx.lineTo(x, y + px); }
          if (foreign(r, c + 1)) { ctx.moveTo(x + px, y); ctx.lineTo(x + px, y + px); }
        }
      }
      ctx.stroke();
      ctx.setLineDash([]);

      if (p.showScenery && cell >= 7) {
        const bob = Math.sin(tick * 0.035) * cell * 0.1;
        const seaAt = (r: number, c: number) =>
          r >= 0 && c >= 0 && r < NR && c < NC && GRID.mask[r][c] === 0;

        // Arabian Sea (west, col ~4, row ~11) and Bay of Bengal (east, col ~24, row ~9)
        stamp(ctx, "ship", ox + 4.5 * cell, screenY(11, cell, oy, 0) + bob, cell / 6);
        stamp(ctx, "ship", ox + 24.5 * cell, screenY(9, cell, oy, 0) - bob, cell / 7);
        ctx.fillStyle = "#467b85";
        ctx.font = `${Math.max(7, cell * 0.4)}px Silkscreen, monospace`;
        ctx.textAlign = "center";
        ctx.fillText("ARABIAN", ox + 4.5 * cell, screenY(8.2, cell, oy, 0));
        ctx.fillText("SEA", ox + 4.5 * cell, screenY(7.55, cell, oy, 0));
        ctx.fillText("BAY OF BENGAL", ox + 24.2 * cell, screenY(6.5, cell, oy, 0));
        ctx.fillText("INDIAN OCEAN", ox + 18.5 * cell, screenY(0.5, cell, oy, 0));

        // compass rose in the southern sea (backend row 3, col 4 -> screen row 28)
        if (seaAt(3, 4)) {
          drawCompassRose(ctx, ox + 4.8 * cell, screenY(3.4, cell, oy, 0), cell * 1.15);
        }

        // sea serpent south of the peninsula (backend row 2, col 16)
        if (seaAt(2, 16)) {
          const s = cell * 0.5;
          const x = ox + 15.8 * cell, y = screenY(2.4, cell, oy, 0) + bob * 0.6;
          ctx.strokeStyle = "rgba(122,47,31,0.85)";
          ctx.lineWidth = Math.max(1.4, s * 0.28);
          for (let k = 0; k < 3; k++) {
            ctx.beginPath();
            ctx.arc(x + k * s * 1.15, y, s * 0.55, Math.PI, 0);
            ctx.stroke();
          }
          ctx.fillStyle = "#7a2f1f";
          ctx.fillRect(x + 3 * s * 1.15 - s * 0.2, y - s * 0.75, s * 0.62, s * 0.62);
          ctx.fillStyle = "#fbf5e4";
          ctx.fillRect(x + 3 * s * 1.15 + s * 0.12, y - s * 0.6, s * 0.16, s * 0.16);
        }
      }

      // ── wind particles (+u10 eastward = +c, +v10 northward = +r) ──
      if (p.showWind && p.timeline && !reducedMotion.matches) {
        const uf = frame(p.timeline[p.view === "prime" ? "prime" : "branched"].u10, p.day);
        const vf = frame(p.timeline[p.view === "prime" ? "prime" : "branched"].v10, p.day);
        ctx.lineWidth = Math.max(1, cell * 0.09);
        for (const pt of particles) {
          const r0 = Math.max(0, Math.min(NR - 1, Math.floor(pt.r)));
          const c0 = Math.max(0, Math.min(NC - 1, Math.floor(pt.c)));
          const i = r0 * NC + c0;
          const u = uf[i], v = vf[i];
          const speed = Math.hypot(u, v);
          const dc = (u / 111) * 0.55, dr = (v / 111) * 0.55;
          const x0 = ox + pt.c * cell + cell / 2;
          const y0 = oy + (NR - 1 - pt.r) * cell + cell / 2;
          pt.c += dc; pt.r += dr; pt.life += 0.012;
          const x1 = ox + pt.c * cell + cell / 2;
          const y1 = oy + (NR - 1 - pt.r) * cell + cell / 2;
          const a = Math.min(0.5, speed / 22) * Math.sin(Math.min(1, pt.life) * Math.PI);
          ctx.strokeStyle = `rgba(122,47,31,${a})`;
          ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
          if (pt.life > 1 || pt.r < 0 || pt.r >= NR || pt.c < 0 || pt.c >= NC) {
            pt.r = Math.random() * NR; pt.c = Math.random() * NC; pt.life = 0;
          }
        }
      }

      // ── painted shock mask ──
      if (p.paintMask.size) {
        for (const [i, w] of p.paintMask) {
          const r = rowOf(i), c = colOf(i);
          const x = ox + c * cell + half, y = screenY(r, cell, oy, half);
          const pulse = 0.55 + 0.45 * Math.sin(tick * 0.08);
          ctx.fillStyle = `rgba(197,61,46,${0.20 + 0.45 * w})`;
          ctx.fillRect(x, y, px, px);
          ctx.strokeStyle = `rgba(232,161,12,${0.35 + 0.45 * w * pulse})`;
          ctx.lineWidth = 1;
          ctx.strokeRect(x + 0.5, y + 0.5, px - 1, px - 1);
        }
      }

      // ── brush preview ──
      const hv = hoverRef.current;
      if (p.paintMode && hv !== null) {
        const hr = rowOf(hv), hc = colOf(hv);
        const R = p.brushRadius;
        ctx.strokeStyle = "rgba(197,61,46,0.75)";
        ctx.lineWidth = 1;
        for (let r = Math.floor(hr - R); r <= Math.ceil(hr + R); r++)
          for (let c = Math.floor(hc - R); c <= Math.ceil(hc + R); c++) {
            if (r < 0 || c < 0 || r >= NR || c >= NC) continue;
            const d = Math.hypot(r - hr, c - hc);
            if (d > R) continue;
            const x = ox + c * cell + half, y = screenY(r, cell, oy, half);
            ctx.globalAlpha = 0.20 + 0.45 * (1 - d / R);
            ctx.strokeRect(x + 0.5, y + 0.5, px - 1, px - 1);
          }
        ctx.globalAlpha = 1;
      }

      // ── sonar ripples ──
      for (const ring of ringRef.current) {
        drawPixelSonarRipple(
          ctx,
          0,
          ring.p,
          ring.kind,
          ring.k,
          Math.max(3, Math.round(cell * 0.34)),
          (dLat, dLon) => {
            const rr = ring.r + dLat;
            const cc = ring.c + dLon;
            return {
              x: ox + cc * cell + cell / 2,
              y: oy + (NR - 1 - rr) * cell + cell / 2,
              visible: rr >= -0.5 && rr <= NR - 0.5 && cc >= -0.5 && cc <= NC - 0.5,
            };
          },
        );
      }

      if (p.showCities) {
        for (const city of GRID.cities) {
          const x = ox + city.c * cell + cell / 2, y = screenY(city.r, cell, oy, 0) + cell / 2;
          stamp(ctx, "house", x, y, cell / 12);
          if (p.showScenery && city.i % 3 === 0) {
            const puff = (tick * 0.012 + city.c / NC) % 1;
            ctx.fillStyle = `rgba(255,248,231,${(1 - puff) * 0.65})`;
            ctx.fillRect(x + cell * 0.15, y - cell * (0.4 + puff * 0.8), 2 + puff * 3, 2 + puff * 3);
          }
        }
      }

      if (p.showScenery) {
        const sceneryOpacity = p.view === "diff" || p.paintMode ? 0.3 : 0.9;
        ctx.globalAlpha = sceneryOpacity;
        drawPixelSun(ctx, ox + NC * cell - 30, Math.max(45, oy - 28));

        for (let k = 0; k < 3; k++) {
          const drift = (tick * 0.001 + k * 0.31) % 1;
          const x = ox + (0.12 + drift * 0.7) * NC * cell;
          const y = oy + (k === 0 ? 2.6 : k === 1 ? 12.5 : 7.5) * cell;
          stamp(ctx, "cloud", x, y, cell / 7);
          if (p.lens === "flood" && k === 1) {
            ctx.fillStyle = "#2f7fb5";
            for (let drop = 0; drop < 5; drop++) {
              const fall = (tick * 0.04 + drop * 0.18) % 1;
              ctx.fillRect(x + (drop - 2) * 6, y + 12 + fall * 22, 2, 5);
            }
          }
        }
        ctx.strokeStyle = "#69442e";
        ctx.lineWidth = 2;
        for (let k = 0; k < 3; k++) {
          const x = ox + ((tick * 0.002 + k * 0.03) % 1) * NC * cell;
          const y = oy + cell * (24 + k * 0.7);
          const flap = Math.sin(tick * 0.12) * 3;
          ctx.beginPath(); ctx.moveTo(x - 5, y - flap); ctx.lineTo(x, y); ctx.lineTo(x + 5, y - flap); ctx.stroke();
        }
        ctx.globalAlpha = 1;
      }

      // ── hover crosshair + bracket ──
      if (hv !== null && !p.paintMode) {
        const hr = rowOf(hv), hc = colOf(hv);
        const x = ox + hc * cell + half, y = screenY(hr, cell, oy, half);
        const g = hoverGlow.current.v;
        ctx.strokeStyle = `rgba(122,47,31,${0.16 * g})`;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(ox, y + px / 2); ctx.lineTo(ox + NC * cell, y + px / 2);
        ctx.moveTo(x + px / 2, oy); ctx.lineTo(x + px / 2, oy + NR * cell);
        ctx.stroke();
        ctx.strokeStyle = `rgba(122,47,31,${0.95 * g})`;
        ctx.lineWidth = 2;
        ctx.strokeRect(x - 1.5, y - 1.5, px + 3, px + 3);
      }

      // ── selected cell bracket ──
      if (p.selected !== null) {
        const sr = rowOf(p.selected), sc = colOf(p.selected);
        const x = ox + sc * cell + half, y = screenY(sr, cell, oy, half);
        const pulse = 0.6 + 0.4 * selPulse.current.v;
        const L = Math.max(3, px * 0.42);
        ctx.strokeStyle = `rgba(178,106,5,${pulse})`;
        ctx.lineWidth = 2;
        const corners: [number, number, number, number][] = [
          [x - 2, y - 2, 1, 1], [x + px + 2, y - 2, -1, 1],
          [x - 2, y + px + 2, 1, -1], [x + px + 2, y + px + 2, -1, -1],
        ];
        for (const [cx, cy, sx, sy] of corners) {
          ctx.beginPath();
          ctx.moveTo(cx + sx * L, cy); ctx.lineTo(cx, cy); ctx.lineTo(cx, cy + sy * L);
          ctx.stroke();
        }
      }

      if (p.running) {
        const t = (tick % 60) / 60;
        const bw = NC * cell * 0.28;
        ctx.fillStyle = "rgba(29,111,163,0.75)";
        ctx.fillRect(ox + t * (NC * cell - bw), oy + NR * cell + 6, bw, 2);
      }

      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [geom, colorFor, showWind]);

  const cellFromEvent = useCallback((e: React.PointerEvent | React.MouseEvent): number | null => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const c = Math.floor((e.clientX - rect.left - geom.ox) / geom.cell);
    const sr = Math.floor((e.clientY - rect.top - geom.oy) / geom.cell);
    const r = NR - 1 - sr;
    if (r < 0 || c < 0 || r >= NR || c >= NC) return null;
    return idx(r, c);
  }, [geom]);

  const brushCells = useCallback((center: number) => brushCellsAround(center, brushRadius), [brushRadius]);

  const handleDown = (e: React.PointerEvent) => {
    const i = cellFromEvent(e);
    if (i === null) return;
    (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
    if (paintMode) {
      paintingRef.current = true;
      eraseRef.current = e.shiftKey || e.button === 2;
      lastCellRef.current = i;
      onPaint(brushCells(i), eraseRef.current);
    } else {
      if (i === selected) {
        spawnCellRipple(rowOf(i), colOf(i), "select");
        onSelect(null);
      } else {
        onSelect(i);
      }
    }
  };

  const handleMove = (e: React.PointerEvent) => {
    const i = cellFromEvent(e);
    setHover(i);
    if (!paintingRef.current || i === null) return;
    const last = lastCellRef.current;
    if (last !== null && last !== i) {
      const r0 = rowOf(last), c0 = colOf(last), r1 = rowOf(i), c1 = colOf(i);
      const steps = Math.max(Math.abs(r1 - r0), Math.abs(c1 - c0));
      for (let s = 1; s <= steps; s++) {
        const rr = Math.round(r0 + ((r1 - r0) * s) / steps);
        const cc = Math.round(c0 + ((c1 - c0) * s) / steps);
        onPaint(brushCells(idx(rr, cc)), eraseRef.current);
      }
    } else {
      onPaint(brushCells(i), eraseRef.current);
    }
    lastCellRef.current = i;
  };

  const handleUp = () => {
    if (paintingRef.current) { paintingRef.current = false; lastCellRef.current = null; onPaintEnd(); }
  };

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

  const pillPos = useMemo(() => {
    if (hover === null) return null;
    const r = rowOf(hover), c = colOf(hover);
    const sr = NR - 1 - r;
    const x = geom.ox + c * geom.cell + geom.cell / 2;
    const y = geom.oy + sr * geom.cell;
    return { x, y, flipX: x > geom.w - 190, flipY: y < 96 };
  }, [hover, geom]);

  return (
    <div ref={wrapRef} className="relative h-full w-full scanlines select-none">
      <canvas
        ref={canvasRef}
        className={paintMode ? "cursor-crosshair touch-none" : "cursor-pointer touch-none"}
        onPointerDown={handleDown}
        onPointerMove={handleMove}
        onPointerUp={handleUp}
        onPointerCancel={handleUp}
        onLostPointerCapture={handleUp}
        onPointerLeave={() => { setHover(null); handleUp(); }}
        onContextMenu={(e) => e.preventDefault()}
      />

      <canvas ref={celebrationRef} aria-hidden="true" className="pointer-events-none absolute inset-0 z-20 h-full w-full" />
      <AnimatePresence>
        {celebrating && (
          <motion.div initial={{ opacity: 0, y: 20, scale: 0.8 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0, y: -20 }} className="pointer-events-none absolute left-1/2 top-14 z-30 -translate-x-1/2 border-2 border-inkline bg-panel px-5 py-3 font-pixel text-xs text-shock shadow-[4px_4px_0_#69442e]">
            A NEW TIMELINE!
          </motion.div>
        )}
      </AnimatePresence>

      {/* corner registration marks — instrument-panel feel */}
      {[
        { left: geom.ox - 7, top: geom.oy - 7, d: "M0 8 L0 0 L8 0" },
        { left: geom.ox + NC * geom.cell - 1, top: geom.oy - 7, d: "M0 0 L8 0 L8 8" },
        { left: geom.ox - 7, top: geom.oy + NR * geom.cell - 1, d: "M0 0 L0 8 L8 8" },
        { left: geom.ox + NC * geom.cell - 1, top: geom.oy + NR * geom.cell - 1, d: "M8 0 L8 8 L0 8" },
      ].map((m, k) => (
        <svg key={k} width="8" height="8" className="pointer-events-none absolute" style={{ left: m.left, top: m.top }}>
          <path d={m.d} stroke="#7a2f1f" strokeWidth="1.5" fill="none" />
        </svg>
      ))}

      {/* axis labels */}
      <div className="pointer-events-none absolute bottom-2 right-2 hidden font-pixel text-[7px] text-ink-faint sm:block">
        {fmtLon(GRID.lon[0])} → {fmtLon(GRID.lon[NC - 1])} · {fmtLat(GRID.lat[NR - 1])} ↓ {fmtLat(GRID.lat[0])} · 1° CELLS
      </div>

      <AnimatePresence>
        {hoverInfo && pillPos && (
          <motion.div
            key="pill"
            initial={{ opacity: 0, scale: 0.94, y: 4 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.96 }}
            transition={{ type: "spring", stiffness: 520, damping: 32 }}
            className="pointer-events-none absolute z-20 w-[196px] panel px-2.5 py-2 text-[10px] leading-tight"
            style={{
              left: pillPos.flipX ? pillPos.x - 202 : pillPos.x + 12,
              top: pillPos.flipY ? pillPos.y + geom.cell + 10 : pillPos.y - 84,
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
              <span className="inline-block size-1.5" style={{ background: hoverInfo.mask === 2 ? "var(--color-safe)" : hoverInfo.mask === 0 ? "#4a9ab8" : "var(--color-ink-faint)" }} />
              {hoverInfo.region}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {!timeline && (
        <div className="absolute inset-0 grid place-items-center">
          <div className="text-center">
            <div className="font-pixel text-[11px] text-ink-dim mm-pulse">AWAITING /simulate</div>
            <div className="mt-2 font-mono text-[10px] text-ink-faint">paint a shock, then branch the timeline</div>
          </div>
        </div>
      )}
    </div>
  );
}
