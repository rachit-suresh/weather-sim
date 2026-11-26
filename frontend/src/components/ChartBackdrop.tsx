import { memo, type ReactElement } from "react";

/* ════════════════════════════════════════════════════════════════════════════
   spec IMP-4h: "explorer's chart" backdrop behind both map views. One static SVG:
   parchment grain + age stains + vignette, a faint graticule, portolan rhumb lines
   from a compass rose, three pixel doodles and a "Here be monsoons" cartouche.
   Low contrast on purpose: it decorates, the map stays the subject.
   ════════════════════════════════════════════════════════════════════════════ */

const INK = "#69442e";

/* pixel doodles: one character = one pixel, "." = empty */
const PAL: Record<string, string> = {
  o: INK, w: "#fbf4e0", B: "#9a6a3e", s: "#c9a46a", g: "#6f8f6a", G: "#8fae84", e: "#2f2116",
  b: "#5b7f95", c: "#88a9bb", r: "#b5523b", "~": "#7fa9b8",
};
const SHIP = [
  "........r.........",
  "........o.........",
  ".....wwwo.wwww....",
  "....wwwwo.wwwww...",
  "...wwwwwo.wwwwww..",
  "..wwwwwwo.wwwwww..",
  "....wwwwo..wwww...",
  "........o.........",
  "ooooooooooooooooo.",
  ".oBBBsBBBBsBBBBo..",
  "..oBBBBBBBBBBBo...",
  "...ooooooooooo....",
  ".~~..~~~..~~~..~~.",
];
const SERPENT = [
  "..gg.....................",
  ".gGgg..........gg........",
  "ggegG.........gGgg....gg.",
  "gg..gg.......gG..gg..gGgg",
  "....gg..ggg..gg...gg.gg..",
  ".....gggGgggg......ggg...",
  "~~~~~~gg~~~~~~~~~~~~~~~~~",
  ".~~~~~~~~~~~~~~~~~~~~~~~.",
];
const WHALE = [
  "......c.c.......",
  ".......c........",
  ".......c........",
  "....bbbbbbb.....",
  "..bbbbbbbbbbb..b",
  ".bebbbbbbbbbbbbb",
  "bbbbbbbbbbbbbbb.",
  ".bccccccbbbbbb..",
  "~~~~~~~~~~~~~~~~",
];

function Sprite({ rows, scale }: { rows: readonly string[]; scale: number }) {
  const rects: ReactElement[] = [];
  rows.forEach((row, y) => {
    let x = 0;
    while (x < row.length) {
      const ch = row[x];
      if (ch === ".") { x++; continue; }
      let run = 1;
      while (x + run < row.length && row[x + run] === ch) run++;
      rects.push(<rect key={`${x}-${y}`} x={x * scale} y={y * scale} width={run * scale} height={scale} fill={PAL[ch]} />);
      x += run;
    }
  });
  return <g shapeRendering="crispEdges">{rects}</g>;
}

function rays(n: number, len: number, phase = 0) {
  return Array.from({ length: n }, (_, k) => {
    const a = phase + (k / n) * Math.PI * 2;
    return <line key={k} x1={0} y1={0} x2={Math.cos(a) * len} y2={Math.sin(a) * len} />;
  });
}

function CompassRose() {
  const point = (angle: number, len: number, half: number, k: number) => {
    const a = (angle * Math.PI) / 180, p = a + Math.PI / 2;
    const tip = [Math.cos(a) * len, Math.sin(a) * len];
    const l = [Math.cos(p) * half, Math.sin(p) * half], r = [-l[0], -l[1]];
    return (
      <g key={k}>
        <polygon points={`0,0 ${tip.join(",")} ${l.join(",")}`} fill="#f6ecd2" stroke={INK} strokeWidth={1} />
        <polygon points={`0,0 ${tip.join(",")} ${r.join(",")}`} fill={INK} stroke={INK} strokeWidth={1} />
      </g>
    );
  };
  return (
    <g>
      <circle r={104} fill="none" stroke={INK} strokeWidth={1.5} />
      <circle r={96} fill="none" stroke={INK} strokeWidth={0.75} />
      {Array.from({ length: 32 }, (_, k) => {
        const a = (k / 32) * Math.PI * 2, r0 = k % 4 === 0 ? 88 : 92;
        return <line key={k} x1={Math.cos(a) * r0} y1={Math.sin(a) * r0} x2={Math.cos(a) * 96} y2={Math.sin(a) * 96} stroke={INK} strokeWidth={1} />;
      })}
      {[45, 135, 225, 315].map((d, k) => point(d, 58, 9, k))}
      {[0, 90, 180, 270].map((d, k) => point(d - 90, 92, 13, k + 4))}
      <circle r={6} fill="#f6ecd2" stroke={INK} strokeWidth={1.5} />
      <text y={-114} textAnchor="middle" className="font-pixel" fontSize={14} fill={INK}>N</text>
    </g>
  );
}

function ChartBackdrop() {
  return (
    <svg className="pointer-events-none absolute inset-0 z-0 h-full w-full" aria-hidden="true">
      <defs>
        <filter id="cb-grain" x="0" y="0" width="100%" height="100%">
          <feTurbulence type="fractalNoise" baseFrequency="0.85" numOctaves={2} stitchTiles="stitch" />
          <feColorMatrix values="0 0 0 0 0.41  0 0 0 0 0.27  0 0 0 0 0.18  0 0 0 0.10 0" />
        </filter>
        <filter id="cb-stains" x="0" y="0" width="100%" height="100%">
          <feTurbulence type="fractalNoise" baseFrequency="0.004" numOctaves={3} seed={7} />
          <feColorMatrix values="0 0 0 0 0.55  0 0 0 0 0.38  0 0 0 0 0.20  0 0 0 0.9 -0.42" />
        </filter>
        <radialGradient id="cb-vignette" cx="50%" cy="48%" r="75%">
          <stop offset="55%" stopColor={INK} stopOpacity={0} />
          <stop offset="100%" stopColor={INK} stopOpacity={0.22} />
        </radialGradient>
        <pattern id="cb-grat" width="72" height="72" patternUnits="userSpaceOnUse">
          <path d="M72 0H0V72" fill="none" stroke={INK} strokeOpacity={0.07} strokeWidth={1} />
        </pattern>
      </defs>

      <rect width="100%" height="100%" filter="url(#cb-stains)" />
      <rect width="100%" height="100%" fill="url(#cb-grat)" />

      {/* portolan rhumb-line nets */}
      <svg x="100%" y="100%" overflow="visible">
        <g transform="translate(-150,-150)" stroke={INK} strokeOpacity={0.11} strokeWidth={1}>{rays(32, 4000)}</g>
      </svg>
      <svg x="9%" y="20%" overflow="visible">
        <g stroke={INK} strokeOpacity={0.07} strokeWidth={1}>{rays(16, 4000, Math.PI / 16)}</g>
        <circle r={4} fill={INK} fillOpacity={0.25} />
      </svg>

      {/* compass rose, bottom-right */}
      <svg x="100%" y="100%" overflow="visible">
        <g transform="translate(-150,-150)" opacity={0.42}><CompassRose /></g>
      </svg>

      {/* doodles */}
      <svg x="13%" y="26%" overflow="visible"><g opacity={0.6}><Sprite rows={SHIP} scale={4} /></g></svg>
      <svg x="80%" y="24%" overflow="visible"><g opacity={0.55}><Sprite rows={WHALE} scale={4} /></g></svg>
      <svg x="74%" y="58%" overflow="visible"><g opacity={0.55}><Sprite rows={SERPENT} scale={4} /></g></svg>

      {/* cartouche */}
      <svg x="3%" y="54%" overflow="visible">
        <g opacity={0.7}>
          <circle cx={0} cy={34} r={12} fill="#f6ecd2" stroke={INK} strokeWidth={1.5} />
          <circle cx={196} cy={34} r={12} fill="#f6ecd2" stroke={INK} strokeWidth={1.5} />
          <rect x={0} y={4} width={196} height={60} fill="#f6ecd2" stroke={INK} strokeWidth={1.5} />
          <rect x={5} y={9} width={186} height={50} fill="none" stroke={INK} strokeWidth={0.75} />
          <text x={98} y={30} textAnchor="middle" className="font-pixel" fontSize={11} fill={INK} letterSpacing={2}>HERE BE</text>
          <text x={98} y={50} textAnchor="middle" className="font-pixel" fontSize={15} fill="#b5523b" letterSpacing={2}>MONSOONS</text>
        </g>
      </svg>

      <rect width="100%" height="100%" filter="url(#cb-grain)" />
      <rect width="100%" height="100%" fill="url(#cb-vignette)" />
    </svg>
  );
}

export default memo(ChartBackdrop);
