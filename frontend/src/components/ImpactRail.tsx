import { useMemo, useState, type CSSProperties } from "react";
import { motion } from "motion/react";
import NumberFlow from "@number-flow/react";
import { CloudRain, Flame, Wheat, ArrowUpRight, ArrowDownRight, Layers } from "lucide-react";
import SmoothTab from "@/components/kokonutui/smooth-tab";
import LiquidGauge from "@/components/kokonutui/liquid-gauge";
import { RadarChart } from "@/components/bklit/charts/radar-chart";
import { RadarGrid } from "@/components/bklit/charts/radar-grid";
import { RadarAxis } from "@/components/bklit/charts/radar-axis";
import { RadarLabels } from "@/components/bklit/charts/radar-labels";
import { RadarArea } from "@/components/bklit/charts/radar-area";
import { SectionTitle, ToggleChip, StatRow, Flap } from "@/components/ui";
import CellInspector from "@/components/CellInspector";
import { computeLens, lensById, LENSES, type LensId } from "@/lib/impact";
import { diffFrame } from "@/lib/api";
import { GRID, INDIA_CELLS, rowOf, colOf, regionOf, nearestCity, displayDelta, displayUnit } from "@/lib/grid";
import { DIFF_FLOOR, diffScale } from "@/lib/colormaps";
import type { Multiverse } from "@/state/useMultiverse";

const LENS_ICONS = { flood: CloudRain, heat: Flame, energy: Wheat } as const;

function placeName(r: number, c: number, lsm?: number) {
  const nc = nearestCity(r, c);
  return nc && nc.distanceCells <= 1.6 ? nc.city.name : regionOf(r, c, lsm);
}

function dedupe(top: { i: number; delta: number; r?: number; c?: number }[], landSeaMask: number[][]) {
  const seen = new Set<string>();
  const rows: { i: number; delta: number; name: string }[] = [];
  for (const cell of top) {
    const r = cell.r ?? rowOf(cell.i);
    const c = cell.c ?? colOf(cell.i);
    const name = placeName(r, c, landSeaMask?.[r]?.[c]);
    if (seen.has(name)) continue;
    seen.add(name);
    rows.push({ i: cell.i, delta: cell.delta, name });
  }
  return rows;
}

export default function ImpactRail({ m }: { m: Multiverse }) {
  const [lastLens, setLastLens] = useState<LensId>("flood");
  const lens = m.lens;
  const t = m.timeline;

  const out = useMemo(
    () => (t && lens ? computeLens(t, lens, m.day) : null),
    [t, lens, m.day],
  );
  const primeRadar = useMemo(
    () => (t && lens ? computeLens({ ...t, branched: t.prime }, lens, m.day).radar : null),
    [t, lens, m.day],
  );

  const rawStats = useMemo(() => {
    if (!t || lens) return null;
    const d = diffFrame(t, m.variable, m.day);
    let max = 0, sum = 0, changed = 0, maxI = 0;
    const eps = m.variable === "msl" ? 1e-2 : 1e-4;
    for (const i of INDIA_CELLS) {
      const a = Math.abs(d[i]);
      sum += a;
      if (a > eps) changed++;
      if (a > max) { max = a; maxI = i; }
    }
    const rawScale = diffScale(d, DIFF_FLOOR, m.variable, GRID.mask.flat().map((cell) => cell === 2 ? 1 : 0));
    const top = INDIA_CELLS
      .map((i) => ({ i, delta: displayDelta(m.variable, d[i]) }))
      .sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta))
      .slice(0, 6);
    return {
      mean: displayDelta(m.variable, sum / INDIA_CELLS.length),
      max: displayDelta(m.variable, max),
      maxI,
      changed,
      scale: displayDelta(m.variable, rawScale),
      top,
    };
  }, [t, lens, m.variable, m.day]);

  const def = lens ? lensById(lens) : null;

  const lensMovers = useMemo(() => (out ? dedupe(out.top, m.landSeaMask) : []), [out, m.landSeaMask]);
  const rawMovers = useMemo(() => (rawStats ? dedupe(rawStats.top, m.landSeaMask) : []), [rawStats, m.landSeaMask]);
  const lensFlat = lensMovers.every((c) => Math.abs(c.delta) < 0.005);
  const rawFlat = rawMovers.every((c) => Math.abs(c.delta) < 0.005);

  return (
    <div className="flex h-full flex-col overflow-y-auto">
      {/* spec IMP-4b: a clicked cell opens this drawer, so its inspector comes first */}
      <CellInspector m={m} />
      <SectionTitle
        right={
          <ToggleChip
            active={lens === null}
            onClick={() => m.setLens(lens === null ? lastLens : null)}
            icon={Layers}
            accent="var(--color-branch)"
          >
            RAW FIELD
          </ToggleChip>
        }
      >
        IMPACT LENS
      </SectionTitle>

      <div className="px-3">
        <SmoothTab
          label="Impact lens"
          value={lens ?? lastLens}
          onChange={(id) => { const l = id as LensId; setLastLens(l); m.setLens(l); }}
          activeColor="bg-amber-400"
          activeTextColor="text-zinc-950"
          className="max-w-none"
          items={LENSES.map((l) => {
            const Icon = LENS_ICONS[l.id];
            return {
              id: l.id,
              title: l.short,
              icon: Icon,
              content: (
                <div className="space-y-2 font-mono text-[10px] leading-relaxed">
                  <div className="font-pixel text-[9px]" style={{ color: l.accent }}>{l.name}</div>
                  <p className="text-ink-dim">{l.tagline}</p>
                  <details className="group border-t border-grid-line pt-1.5">
                    <summary className="cursor-pointer select-none font-pixel text-[8px] text-ink-faint transition-colors hover:text-ink">
                      MODEL CARD <span className="inline-block transition-transform group-open:rotate-90">▸</span>
                    </summary>
                    <div className="mt-1.5 space-y-1.5">
                      <div>
                        <div className="font-pixel text-[8px] text-ink-faint">FORMULA</div>
                        <p className="mt-0.5 break-words text-[9px] text-ink">{l.formula}</p>
                      </div>
                      <div>
                        <div className="font-pixel text-[8px] text-ink-faint">THRESHOLDS</div>
                        <p className="mt-0.5 text-[9px] text-ink-dim">{l.thresholds}</p>
                      </div>
                      <div>
                        <div className="font-pixel text-[8px] text-branch">PROXY CAVEATS</div>
                        <p className="mt-0.5 text-[9px] text-ink-faint">{l.caveats}</p>
                      </div>
                      <div className="text-[9px] text-ink-faint">
                        inputs: {l.inputs.join(" · ")}
                      </div>
                    </div>
                  </details>
                </div>
              ),
            };
          })}
        />
      </div>

      {/* ── live lens output ── */}
      {lens && out && def && (
        <div className="px-3 pt-3">
          <div className="flex items-start gap-3">
            <div className="relative shrink-0 overflow-hidden" style={{ width: 159, height: 198 }}>
              <div className="origin-top-left" style={{ transform: "scale(0.55)", "--gauge-accent": def.accent } as CSSProperties}>
                <LiquidGauge
                  value={out.severity}
                  readOnly
                  label="SEVERITY"
                  total={100}
                  unit="/100"
                  sublineTail="/ 100 index"
                  statusLabel="ALERT AT 70"
                  warnAt={70}
                  liquidClass="bg-(--gauge-accent) text-void"
                  className="border-grid-line bg-abyss"
                />
              </div>
            </div>
            <div className="min-w-0 flex-1 space-y-0.5 pt-1">
              <div className="font-pixel text-[8px] text-ink-faint">DAY {m.day + 1} STATUS</div>
              <StatRow label="Alert area" value={(out.areaKm2 / 1000).toFixed(0)} unit="k km²" tone={def.accent} />
              <StatRow label="Pop-wtd exposure" value={out.exposure.toFixed(1)} unit="/100" />
              <StatRow label="Cells on alert" value={`${out.cellsAlert}`} unit={`/ ${INDIA_CELLS.length}`} />
              <StatRow label="Diff scale" value={`±${out.scale.toFixed(2)}`} unit={def.unit} />
            </div>
          </div>
          <Flap text={out.severityLabel} length={11} scale={0.5} className="mt-2" />

          <div className="mt-3 grid grid-cols-2 gap-1.5">
            {out.metrics.map((met) => (
              <div key={met.label} className="border border-grid-line bg-abyss/70 px-2 py-1.5" title={met.hint}>
                <div className="truncate font-pixel text-[7px] text-ink-faint">{met.label.toUpperCase()}</div>
                <div className="mt-0.5 flex items-baseline gap-1">
                  <NumberFlow
                    value={met.value}
                    format={{ minimumFractionDigits: met.precision ?? 1, maximumFractionDigits: met.precision ?? 1 }}
                    className="font-mono text-[15px] font-medium text-ink"
                  />
                  <span className="text-[8px] text-ink-faint">{met.unit}</span>
                </div>
                {met.delta !== undefined && Math.abs(met.delta) > 1e-3 && (
                  <div className={`mt-0.5 flex items-center gap-0.5 font-mono text-[9px] ${met.delta > 0 ? "text-branch" : "text-prime"}`}>
                    {met.delta > 0 ? <ArrowUpRight size={9} /> : <ArrowDownRight size={9} />}
                    {met.delta > 0 ? "+" : ""}{met.delta.toFixed(met.precision ?? 1)} vs prime
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* radar fingerprint */}
          <div className="mt-3 border border-grid-line bg-abyss/50">
            <div className="flex items-center justify-between px-2 pt-2">
              <span className="font-pixel text-[8px] text-ink-faint">RISK FINGERPRINT</span>
              <span className="flex items-center gap-2 font-mono text-[8px] text-ink-faint">
                <span className="flex items-center gap-1"><i className="inline-block size-1.5" style={{ background: def.accent }} />branch</span>
                <span className="flex items-center gap-1"><i className="inline-block size-1.5 bg-prime" />prime</span>
              </span>
            </div>
            <RadarChart
              data={[
                { label: "Branched", color: def.accent, values: out.radar },
                ...(primeRadar ? [{ label: "Prime", color: "#1d6fa3", values: primeRadar }] : []),
              ]}
              metrics={Object.keys(out.radar).map((k) => ({ key: k, label: k }))}
              size={252}
              levels={4}
              margin={52}
              className="mx-auto"
            >
              <RadarGrid />
              <RadarAxis />
              <RadarLabels />
              <RadarArea index={1} color="#1d6fa3" showPoints={false} />
              <RadarArea index={0} color={def.accent} />
            </RadarChart>
          </div>

          {/* top movers */}
          <div className="mt-3">
            <div className="font-pixel text-[8px] text-ink-faint">TOP MOVERS · Δ vs PRIME</div>
            <div className="mt-1 space-y-0.5">
              {lensFlat ? (
                <p className="border border-grid-line bg-abyss/50 px-2 py-1.5 text-[9px] text-ink-faint">
                  No movers yet — branch a shock to rank regions.
                </p>
              ) : lensMovers.map((c, k) => (
                <motion.button
                  key={c.i}
                  initial={{ opacity: 0, x: 8 }} animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: k * 0.04 }}
                  onClick={() => { m.selectCell(c.i); m.setView("diff"); }}
                  className={`flex w-full items-center justify-between gap-2 border px-2 py-1 text-left transition-colors ${
                    m.selected === c.i ? "border-branch/60 bg-branch/10" : "border-grid-line bg-abyss/50 hover:border-branch/40"
                  }`}
                >
                  <span className="truncate font-mono text-[10px] text-ink-dim">{c.name}</span>
                  <span className={`font-mono text-[10px] tabular-nums ${c.delta > 0 ? "text-branch" : "text-prime"}`}>
                    {c.delta > 0 ? "+" : ""}{c.delta.toFixed(2)} {def.unit}
                  </span>
                </motion.button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ── raw-field divergence ── */}
      {!lens && t && !m.diverged && (
        <p className="mx-3 mt-3 border border-dashed border-grid-line px-2.5 py-2 text-[10px] leading-relaxed text-ink-dim">
          Nothing to compare yet. Open <span className="font-pixel text-[8px] text-shock">SHOCKS</span>, pick one and press{" "}
          <span className="font-pixel text-[8px] text-shock">BRANCH TIMELINE</span>; how far the branch drifts from Prime shows up here.
        </p>
      )}
      {!lens && rawStats && t && m.diverged && (
        <div className="px-3 pt-3">
          <div className="border border-grid-line bg-abyss/70 px-2.5 py-2">
            <div className="font-pixel text-[8px] text-ink-faint">BRANCH DIVERGENCE · {m.variable.toUpperCase()}</div>
            <div className="mt-1.5 space-y-0.5">
              <StatRow label="Mean |Δ| over India" value={rawStats.mean.toFixed(3)} unit={displayUnit(m.variable)} />
              <StatRow label="Max |Δ|" value={rawStats.max.toFixed(3)} unit={displayUnit(m.variable)} tone="var(--color-branch)" />
              <StatRow label="Cells diverged" value={`${rawStats.changed}`} unit={`/ ${INDIA_CELLS.length}`} />
              <StatRow label="Diff-map scale" value={`±${rawStats.scale.toFixed(2)}`} unit={displayUnit(m.variable)} />
            </div>
            {!m.diverged && (
              <p className="mt-1.5 border-t border-grid-line pt-1.5 text-[9px] text-ink-faint">
                Timelines identical — arm a shock and branch to see divergence.
              </p>
            )}
          </div>
          <div className="mt-2">
            <div className="font-pixel text-[8px] text-ink-faint">TOP MOVERS</div>
            <div className="mt-1 space-y-0.5">
              {rawFlat ? (
                <p className="border border-grid-line bg-abyss/50 px-2 py-1.5 text-[9px] text-ink-faint">
                  No movers yet — branch a shock to rank regions.
                </p>
              ) : rawMovers.map((c) => (
                <button
                  key={c.i}
                  onClick={() => m.selectCell(c.i)}
                  className={`flex w-full items-center justify-between gap-2 border px-2 py-1 text-left transition-colors ${
                    m.selected === c.i ? "border-branch/60 bg-branch/10" : "border-grid-line bg-abyss/50 hover:border-branch/40"
                  }`}
                >
                  <span className="truncate font-mono text-[10px] text-ink-dim">{c.name}</span>
                  <span className={`font-mono text-[10px] tabular-nums ${c.delta > 0 ? "text-branch" : "text-prime"}`}>
                    {c.delta > 0 ? "+" : ""}{c.delta.toFixed(2)} {displayUnit(m.variable)}
                  </span>
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {!t && (
        <div className="px-3 pt-4 text-center font-pixel text-[9px] text-ink-faint mm-pulse">
          AWAITING TIMELINE…
        </div>
      )}

      <div className="h-4 shrink-0" />
    </div>
  );
}
