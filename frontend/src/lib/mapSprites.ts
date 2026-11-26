import {
  CELLS, GRID, NC, NR, colOf, rowOf, idx, isIndia, type VarKey,
} from "@/lib/grid";
import { frame, diffFrame, type Timeline } from "@/lib/api";
import {
  ABS_RANGE, SEQUENTIAL, divergingColor, moistColor, HEAT_RAMP, ENERGY_RAMP,
  diffScale, DIFF_FLOOR, type RGB,
} from "@/lib/colormaps";
import type { LensId, LensOutput } from "@/lib/impact";
import type { ViewMode } from "@/components/PixelGrid";

export const PAPER: RGB = [243, 233, 207];

/* ── storybook terrain: one biome per cell, baked once from mask + elevation ── */
export type Biome = "ocean" | "foreign" | "grass" | "forest" | "sand" | "mountain" | "highland";

export const BIOME_BASE: Record<Biome, RGB> = {
  ocean: [143, 208, 222],
  foreign: [217, 196, 143],
  grass: [124, 184, 92],
  forest: [76, 139, 63],
  sand: [236, 195, 111],
  mountain: [203, 184, 224],
  highland: [176, 156, 108],
};

export const cellHash = (r: number, c: number) => ((r * 73856093) ^ (c * 19349663)) >>> 0;

export const TERRAIN: { biome: Biome; base: RGB; v: number }[] = (() => {
  const out: { biome: Biome; base: RGB; v: number }[] = [];
  for (let r = 0; r < NR; r++) {
    for (let c = 0; c < NC; c++) {
      const mask = GRID.mask[r][c];
      const elev = GRID.elevation[r][c];
      const lat = GRID.lat[r], lon = GRID.lon[c];
      let biome: Biome;
      if (mask === 0) biome = "ocean";
      else if (elev >= 0.6) biome = "mountain";
      else if (mask === 1) biome = "foreign";
      else if (lat >= 23 && lat <= 31.5 && lon >= 68 && lon <= 76.5 && elev < 0.2) biome = "sand";
      else if (elev >= 0.25) biome = "highland";
      else if (elev >= 0.12 || (lat > 24 && lon > 89)) biome = "forest";
      else biome = "grass";
      const h = cellHash(r, c);
      const v = h % 4;
      const b = BIOME_BASE[biome];
      const k = 1 + (v - 1.5) * 0.028;
      out.push({ biome, base: [b[0] * k, b[1] * k, b[2] * k], v });
    }
  }
  return out;
})();

export const SPRITES = {
  tree: ["...oo...", "..oggo..", ".oggggo.", "ogGggggo", "oggggggo", ".oggggo.", "...tt...", "..tttt.."],
  pine: ["....o....", "...ogo...", "..ogGgo..", "...ogo...", "..oggggo.", ".ogGgggo.", "ogggggggo", "...ttt...", "...ttt..."],
  peak: [".....o.....", "....owo....", "...owwwo...", "..owwwWwo..", "..owWWWWo..", ".ovvWVVVVo.", "ovvvVVVVVVo", "ooooooooooo"],
  cloud: ["....ooo.......", "..ooWWWoo.....", ".oWWWWWWWo....", "oWWWWWWWWWooo.", "oWWWWWWWWWWWWo", ".oWccccccccco.", "..oooooooooo.."],
  ship: ["......r.......", "......rrrr....", "......t.......", "......twww....", "......twwww...", "......twwww...", "......twww....", "..o...t.......", "otttttttttttto", ".otttttttttto.", "..oooooooooo.."],
  house: ["....rr....", "...rrrr...", "..rrrrrr..", ".rrrrrrrr.", "oooooooooo", ".owwBwwwo.", ".owwwwwwo.", ".owwttwwo.", ".owwttwwo.", ".oooooooo."],
  cactus: ["..g...g.", "..g.g.g.", "..ggggg.", "....g...", ".g..g...", ".gggg...", "....g...", "...ttt.."],
  palm: ["ggg..ggg", ".gggggg.", "gggggggg", "...t....", "...t....", "....t...", "....t...", "...ttt.."],
} as const;

export const SPRITE_INK: Record<string, string> = {
  o: "#69442e", g: "#438238", G: "#a5cf68", t: "#91522c", r: "#ce4d34",
  w: "#fff8e7", W: "#fffdf3", c: "#d5ded9", v: "#b2a1c9", V: "#8c7ca7", B: "#529db5",
};

/** Stamp a pixel sprite centered at (x, y) with pixel size rounded from unit. */
export function stamp(ctx: CanvasRenderingContext2D, name: keyof typeof SPRITES, x: number, y: number, unit: number) {
  const pixels = SPRITES[name];
  const s = Math.max(1, Math.round(unit));
  const left = Math.round(x - (pixels[0].length * s) / 2);
  const top = Math.round(y - (pixels.length * s) / 2);
  for (let row = 0; row < pixels.length; row++) {
    const line = pixels[row];
    for (let col = 0; col < line.length; col++) {
      const color = line[col];
      if (color === ".") continue;
      ctx.fillStyle = SPRITE_INK[color];
      ctx.fillRect(left + col * s, top + row * s, s, s);
    }
  }
}

/** Shared target field extraction for PixelGrid and GlobeView. */
export function computeTargetField(
  timeline: Timeline | null,
  view: ViewMode,
  variable: VarKey,
  day: number,
  lens: LensId | null,
  lensOut: LensOutput | null,
) {
  if (!timeline) return { values: new Float32Array(CELLS), scale: 1, kind: "none" as const };
  if (lens && lensOut) {
    const vals = view === "prime" ? lensOut.base : view === "branched" ? lensOut.value : lensOut.delta;
    return { values: vals, scale: lensOut.scale, kind: "lens" as const };
  }
  if (view === "diff") {
    const d = diffFrame(timeline, variable, day);
    const indiaMask = new Array<number>(CELLS);
    for (let i = 0; i < CELLS; i++) indiaMask[i] = isIndia(rowOf(i), colOf(i)) ? 1 : 0;
    return { values: d, scale: diffScale(d, DIFF_FLOOR, variable, indiaMask), kind: "diff" as const };
  }
  return { values: frame(timeline[view][variable], day), scale: 1, kind: "abs" as const };
}

/** Shared cell colour mapping for current variable / view / lens. */
export function colorForFieldValue(
  v: number,
  scaleV: number,
  view: ViewMode,
  variable: VarKey,
  lens: LensId | null,
  lensOut: LensOutput | null,
): RGB {
  if (lens && lensOut) {
    if (view === "diff") {
      const s = scaleV || 1;
      const t = Math.max(-1, Math.min(1, v / s));
      return lens === "flood" ? moistColor(t) : lens === "energy" ? ENERGY_RAMP((t + 1) / 2) : divergingColor(t);
    } else if (lens === "heat") {
      return HEAT_RAMP(Math.max(0, Math.min(1, (v - 22) / 13)));
    } else if (lens === "flood") {
      return SEQUENTIAL.tp(Math.max(0, Math.min(1, v / 100)));
    }
    return ENERGY_RAMP(Math.max(0, Math.min(1, (v + 6) / 16)));
  } else if (view === "diff") {
    const s = scaleV || 1;
    return divergingColor(Math.max(-1, Math.min(1, v / s)));
  }
  const [lo, hi] = ABS_RANGE[variable];
  return SEQUENTIAL[variable](Math.max(0, Math.min(1, (v - lo) / (hi - lo))));
}

/**
 * Storybook overlay strength (spec IMP-3d): how strongly a forecast value tints the art map.
 * Strong where the value is notable (wet, changed, alert), faint where it is ordinary, so the terrain stays readable.
 * mask: 0 sea, 1 neighbour land, 2 India.
 */
export function storyAlpha(
  v: number, scaleV: number, view: ViewMode, variable: VarKey, lens: LensId | null, mask: number,
): number {
  const clamp = (x: number, a: number, b: number) => Math.max(a, Math.min(b, x));
  let a: number;
  // diff colours are near-white for small changes: give them no tint (no washed-out art), then ramp steeply
  if (view === "diff") { const m = Math.abs(v) / (scaleV || 1); return m < 0.08 ? 0 : clamp((m - 0.08) * 3.4, 0.25, 0.9); }
  else if (lens === "flood") a = clamp((v - 20) / 50, 0, 0.72);
  else if (lens === "heat") a = clamp((v - 23) / 9, 0, 0.72);
  else if (lens === "energy") a = clamp(Math.abs(v) / 8, 0, 0.6);
  else if (variable === "tp") a = v < 1 ? 0 : clamp(v / 30, 0.12, 0.75);
  else a = 0.34;
  return a * (mask === 0 ? 0.7 : mask === 1 ? 0.8 : 1);
}

/** Gaussian shock brush kernel around a grid cell. */
export function brushCellsAround(center: number, brushRadius: number): { i: number; w: number }[] {
  const hr = rowOf(center), hc = colOf(center);
  const R = brushRadius;
  const out: { i: number; w: number }[] = [];
  for (let r = Math.floor(hr - R); r <= Math.ceil(hr + R); r++) {
    for (let c = Math.floor(hc - R); c <= Math.ceil(hc + R); c++) {
      if (r < 0 || c < 0 || r >= NR || c >= NC) continue;
      const d = Math.hypot(r - hr, c - hc);
      if (d > R) continue;
      const w = R <= 0.6 ? 1 : Math.exp(-0.5 * (d / (R / 1.7)) ** 2);
      if (w > 0.05) out.push({ i: idx(r, c), w: Math.min(1, w) });
    }
  }
  return out;
}

/** Draw the cheeky storybook pixel-art sun centered at (sunX, sunY). */
export function drawPixelSun(ctx: CanvasRenderingContext2D, sunX: number, sunY: number, tick = 0, withRays = false) {
  if (withRays) {
    const rayAlpha = 0.28 + 0.16 * Math.sin(tick * 0.035);
    ctx.strokeStyle = `rgba(192, 122, 8, ${rayAlpha})`;
    ctx.lineWidth = 2;
    for (let a = 0; a < 8; a++) {
      const ang = (a / 8) * Math.PI * 2 + tick * 0.002;
      ctx.beginPath();
      ctx.moveTo(sunX + Math.cos(ang) * 18, sunY + Math.sin(ang) * 18);
      ctx.lineTo(sunX + Math.cos(ang) * 25, sunY + Math.sin(ang) * 25);
      ctx.stroke();
    }
  }
  ctx.fillStyle = "#b77918";
  ctx.fillRect(sunX - 13, sunY - 10, 26, 20);
  ctx.fillRect(sunX - 10, sunY - 13, 20, 26);
  ctx.fillStyle = "#f4bd35";
  ctx.fillRect(sunX - 11, sunY - 8, 22, 16);
  ctx.fillRect(sunX - 8, sunY - 11, 16, 22);
  ctx.fillStyle = "#fff2a6";
  ctx.fillRect(sunX - 7, sunY - 7, 6, 5);
  ctx.fillStyle = "#69442e";
  ctx.fillRect(sunX - 5, sunY - 1, 2, 3);
  ctx.fillRect(sunX + 4, sunY - 1, 2, 3);
  ctx.fillRect(sunX - 2, sunY + 5, 6, 2);
}

/** Draw the maroon-ink compass rose centered at (cx, cy) with radius s. */
export function drawCompassRose(ctx: CanvasRenderingContext2D, cx: number, cy: number, s: number) {
  ctx.strokeStyle = "rgba(122,47,31,0.85)";
  ctx.lineWidth = 1.2;
  ctx.beginPath(); ctx.arc(cx, cy, s, 0, Math.PI * 2); ctx.stroke();
  ctx.beginPath(); ctx.arc(cx, cy, s * 0.62, 0, Math.PI * 2); ctx.stroke();
  const point = (ang: number, len: number, col: string) => {
    ctx.fillStyle = col;
    ctx.beginPath();
    ctx.moveTo(cx + Math.cos(ang) * len, cy + Math.sin(ang) * len);
    ctx.lineTo(cx + Math.cos(ang + 0.5) * len * 0.22, cy + Math.sin(ang + 0.5) * len * 0.22);
    ctx.lineTo(cx + Math.cos(ang - 0.5) * len * 0.22, cy + Math.sin(ang - 0.5) * len * 0.22);
    ctx.closePath();
    ctx.fill();
  };
  point(-Math.PI / 2, s * 0.95, "#c53d2e");
  point(Math.PI / 2, s * 0.95, "#7a2f1f");
  point(0, s * 0.95, "#7a2f1f");
  point(Math.PI, s * 0.95, "#7a2f1f");
  ctx.fillStyle = "#e8a10c";
  ctx.fillRect(cx - s * 0.09, cy - s * 0.09, s * 0.18, s * 0.18);
  ctx.fillStyle = "#7a2f1f";
  ctx.font = `${Math.max(7, s * 0.42)}px Silkscreen, monospace`;
  ctx.textAlign = "center";
  ctx.fillText("N", cx, cy - s * 1.18);
}

const RIPPLE_COLORS: Record<"shock" | "run" | "select", { ink: string; ringA: string; ringB: string; spark: string }> = {
  select: { ink: "74,28,16", ringA: "255,213,61", ringB: "232,90,56", spark: "255,253,243" }, // Coin-gold + coral-orange + cream flash
  shock:  { ink: "74,28,16", ringA: "224,62,45", ringB: "255,196,46", spark: "255,248,220" }, // Brick-red + gold shockwave
  run:    { ink: "32,21,9",  ringA: "45,152,218", ringB: "164,226,245", spark: "255,253,243" }, // Cerulean + sky pulse
};

/**
 * Draw a crisp continuous pixel-art sonar wave + spark burst at an arbitrary degree offset projector.
 * Alpha stays vivid through expansion before dissolving, with dark-ink pixel borders for high contrast.
 */
export function drawPixelSonarRipple(
  ctx: CanvasRenderingContext2D,
  centerLatDeg: number,
  progress: number,
  kind: "shock" | "run" | "select",
  ringIdx: number,
  pipSize: number,
  projectOffset: (dLat: number, dLon: number) => { x: number; y: number; visible: boolean },
) {
  if (progress <= 0.001 || progress >= 1) return;

  // Hold full crisp opacity while expanding cleanly, then smoothly dissolve over the tail 42%
  const rawFade = progress < 0.58 ? 1 : Math.max(0, 1 - (progress - 0.58) / 0.42);
  const ringAlpha = Math.min(1, rawFade * (ringIdx === 0 ? 1.0 : ringIdx === 1 ? 0.9 : 0.72));
  if (ringAlpha <= 0.02) return;

  const pal = RIPPLE_COLORS[kind];
  const maxDeg = kind === "shock" ? 13.5 : kind === "run" ? 10.0 : 6.8;
  const rad = Math.max(0.35, progress * maxDeg);
  const pip = Math.max(3, Math.round(pipSize));
  const cosLat = Math.max(0.35, Math.cos((centerLatDeg * Math.PI) / 180));

  const steps = Math.max(64, Math.min(144, Math.round(rad * 18)));
  const seen = new Set<string>();
  const pts: { x: number; y: number }[] = [];
  for (let s = 0; s < steps; s++) {
    const ang = (s / steps) * Math.PI * 2;
    const pt = projectOffset(Math.sin(ang) * rad, (Math.cos(ang) * rad) / cosLat);
    if (!pt.visible) continue;
    const qx = Math.round(pt.x / pip) * pip;
    const qy = Math.round(pt.y / pip) * pip;
    const key = `${qx},${qy}`;
    if (seen.has(key)) continue;
    seen.add(key);
    pts.push({ x: qx, y: qy });
  }

  if (pts.length) {
    // Pass 1: crisp 1.5px dark maroon ink border around every ring pixel block
    ctx.fillStyle = `rgba(${pal.ink},${(ringAlpha * 0.92).toFixed(3)})`;
    for (const q of pts) ctx.fillRect(q.x - 1, q.y - 1, pip + 2, pip + 2);

    // Pass 2: high-contrast bright color (ring 1 alternates accent band for multi-wave richness)
    const fillRgb = ringIdx === 1 ? pal.ringB : pal.ringA;
    ctx.fillStyle = `rgba(${fillRgb},${ringAlpha.toFixed(3)})`;
    for (const q of pts) ctx.fillRect(q.x, q.y, pip, pip);

    // Pass 3: subtle specular top-left gleam on Ring 0
    if (ringIdx === 0 && pip >= 3) {
      ctx.fillStyle = `rgba(${pal.spark},${(ringAlpha * 0.7).toFixed(3)})`;
      for (const q of pts) ctx.fillRect(q.x, q.y, Math.max(1, pip - 2), Math.max(1, pip - 2));
    }
  }

  // Epicenter pop flash + 4 racing pixel sparks on Ring 0
  if (ringIdx === 0 && progress < 0.72) {
    const burstFade = Math.min(1, (0.72 - progress) / 0.55);
    if (progress < 0.45) {
      const cPt = projectOffset(0, 0);
      if (cPt.visible) {
        const s = Math.max(4, Math.round(pip * (2.4 - progress * 2.5)));
        const fx = Math.round(cPt.x - s / 2);
        const fy = Math.round(cPt.y - s / 2);
        ctx.fillStyle = `rgba(${pal.ink},${burstFade.toFixed(3)})`;
        ctx.fillRect(fx - 1, fy - 1, s + 2, s + 2);
        ctx.fillStyle = `rgba(${pal.spark},${burstFade.toFixed(3)})`;
        ctx.fillRect(fx, fy, s, s);
      }
    }

    const sparkR = rad * 0.85;
    const spSize = Math.max(3, Math.round(pip * 0.95));
    for (let k = 0; k < 4; k++) {
      const ang = (k * Math.PI) / 2 + Math.PI / 4;
      const pt = projectOffset(Math.sin(ang) * sparkR, (Math.cos(ang) * sparkR) / cosLat);
      if (!pt.visible) continue;
      const sx = Math.round(pt.x - spSize / 2);
      const sy = Math.round(pt.y - spSize / 2);
      ctx.fillStyle = `rgba(${pal.ink},${burstFade.toFixed(3)})`;
      ctx.fillRect(sx - 1, sy - 1, spSize + 2, spSize + 2);
      ctx.fillStyle = `rgba(${pal.spark},${burstFade.toFixed(3)})`;
      ctx.fillRect(sx, sy, spSize, spSize);
    }
  }
}
