import { useMemo } from "react";
import { motion } from "motion/react";
import { X, MapPin } from "lucide-react";
import { LineSparkline } from "@/components/bklit/charts/sparkline";
import { StatRow } from "@/components/ui";
import { cellDiagnostics } from "@/lib/impact";
import { cellSeries } from "@/lib/api";
import { GRID, rowOf, colOf, fmtLat, fmtLon, regionOf, nearestCity, display, displayUnit, VAR_META } from "@/lib/grid";
import type { Multiverse } from "@/state/useMultiverse";

export default function CellInspector({ m }: { m: Multiverse }) {
  const t = m.timeline;
  const i = m.selected;
  const src = m.view === "prime" ? "prime" : "branched";

  const diag = useMemo(() => (t && i !== null ? cellDiagnostics(t, m.day, i, src) : null), [t, i, m.day, src]);
  const series = useMemo(() => {
    if (!t || i === null) return null;
    const v = m.variable;
    return {
      prime: cellSeries(t.prime[v], i, t.days).map((x, d) => ({ d, v: display(v, x) })),
      branched: cellSeries(t.branched[v], i, t.days).map((x, d) => ({ d, v: display(v, x) })),
    };
  }, [t, i, m.variable]);

  if (i === null || !t || !diag || !series) return null;

  const r = rowOf(i), c = colOf(i);
  const lsm = m.landSeaMask?.[r]?.[c];
  const nc = nearestCity(r, c);
  const title = nc && nc.distanceCells <= 1.6 ? nc.city.name : `CELL ${r},${c}`;
  const unit = displayUnit(m.variable);

  return (
    <motion.div
      initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }}
      className="mx-3 mt-3 border border-branch/40 bg-abyss/80 glow-branch"
    >
      <div className="flex items-center justify-between border-b border-grid-line px-2.5 py-1.5">
        <span className="flex items-center gap-1.5 font-pixel text-[9px] text-branch">
          <MapPin size={10} /> {title.toUpperCase()}
        </span>
        <button onClick={() => m.selectCell(null)} className="text-ink-faint hover:text-ink" aria-label="Close inspector">
          <X size={12} />
        </button>
      </div>

      <div className="px-2.5 py-2">
        <div className="flex items-baseline justify-between font-mono text-[9px] text-ink-faint">
          <span>{fmtLat(GRID.lat[r])} {fmtLon(GRID.lon[c])}</span>
          <span>{regionOf(r, c, lsm)}</span>
        </div>

        <div className="mt-2 grid grid-cols-2 gap-x-3">
          <StatRow label="Temp" value={diag.tC.toFixed(1)} unit="°C" />
          <StatRow label="Rain" value={diag.tp.toFixed(1)} unit="mm/d" />
          <StatRow label="MSL" value={diag.mslHpa.toFixed(1)} unit="hPa" />
          <StatRow label="RH (proxy)" value={diag.rh.toFixed(0)} unit="%" />
          <StatRow label="Wet-bulb" value={diag.tw.toFixed(1)} unit="°C" tone={diag.tw >= 31 ? "var(--color-danger)" : undefined} />
          <StatRow label="FRI" value={diag.fri.toFixed(0)} tone={diag.fri >= 55 ? "var(--color-danger)" : undefined} />
          <StatRow label="Water bal." value={diag.waterBalance.toFixed(1)} unit="mm/d" tone={diag.waterBalance < -2 ? "var(--color-branch)" : undefined} />
          <StatRow label="Wind₁₀₀" value={diag.v100.toFixed(1)} unit="m/s" />
          <StatRow label="WPD" value={diag.wpd.toFixed(0)} unit="W/m²" />
          <StatRow label="Wind dir" value={`${diag.windFromCompass} (${diag.windFromDeg.toFixed(0)}°)`} />
        </div>
        <div className="mt-1 font-mono text-[9px] text-ink-faint">
          IMD class: <span className="text-ink-dim">{diag.imd}</span> · cell {diag.areaKm2.toFixed(0)} km²
        </div>

        <div className="mt-2 border-t border-grid-line pt-2">
          <div className="flex items-center justify-between">
            <span className="font-pixel text-[8px] text-ink-faint">{VAR_META[m.variable].short} · {t.days} DAYS</span>
            <span className="font-mono text-[8px] text-ink-faint">{unit}</span>
          </div>
          <div className="mt-1 space-y-1">
            <div className="flex items-center gap-2">
              <span className="w-12 shrink-0 font-mono text-[8px] text-prime">PRIME</span>
              <div className="min-w-0 flex-1">
                <LineSparkline data={series.prime} dataKey="v" height={26} color="#1d6fa3" endDot padding={4} />
              </div>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-12 shrink-0 font-mono text-[8px] text-branch">BRANCH</span>
              <div className="min-w-0 flex-1">
                <LineSparkline data={series.branched} dataKey="v" height={26} color="#c07a08" endDot padding={4} />
              </div>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
