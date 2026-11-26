import { useMemo, useState } from "react";
import { Play, Pause, SkipBack, LineChart as ChartIcon, ChevronUp, ChevronDown } from "lucide-react";
import { curveMonotoneX } from "d3-shape";
import { Flap, PixelScrubber } from "@/components/ui";
import { LineChart } from "@/components/bklit/charts/line-chart";
import { Line } from "@/components/bklit/charts/line";
import { Grid } from "@/components/bklit/charts/grid";
import { XAxis } from "@/components/bklit/charts/x-axis";
import { YAxis } from "@/components/bklit/charts/y-axis";
import { ChartTooltip } from "@/components/bklit/charts/chart-tooltip";
import { ReferenceArea } from "@/components/bklit/charts/reference-area";
import { frame } from "@/lib/api";
import { INDIA_CELLS, display, displayUnit } from "@/lib/grid";
import { lensSeries, lensById } from "@/lib/impact";
import type { Multiverse } from "@/state/useMultiverse";

const YEAR_START = Date.UTC(2020, 0, 1);

export default function TimelineDeck({ m }: { m: Multiverse }) {
  const t = m.timeline;
  const [chartOpen, setChartOpen] = useState(false); // spec IMP-4d: chart folded by default
  const doy = t?.dayOfYear ?? 196;
  const baseMs = YEAR_START + doy * 86400000;
  const dayDate = (d: number) => new Date(baseMs + d * 86400000);

  const data = useMemo(() => {
    if (!t) return [];
    if (m.lens) {
      const s = lensSeries(t, m.lens, "value");
      return s.map((p, d) => ({ date: dayDate(d), Prime: +p.prime.toFixed(3), Branched: +p.branched.toFixed(3) }));
    }
    const v = m.variable;
    const out: { date: Date; Prime: number; Branched: number }[] = [];
    for (let d = 0; d < t.days; d++) {
      const pf = frame(t.prime[v], d), bf = frame(t.branched[v], d);
      let sp = 0, sb = 0;
      for (const i of INDIA_CELLS) { sp += pf[i]; sb += bf[i]; }
      const n = INDIA_CELLS.length;
      out.push({ date: dayDate(d), Prime: +display(v, sp / n).toFixed(3), Branched: +display(v, sb / n).toFixed(3) });
    }
    return out;
  }, [t, m.lens, m.variable, baseMs]);

  const unit = m.lens ? lensById(m.lens).unit : displayUnit(m.variable);
  const title = m.lens ? `${lensById(m.lens).short} INDEX` : `${m.variable.toUpperCase()} · INDIA MEAN`;

  return (
    <div className="shrink-0 border-t border-grid-line bg-abyss/85 backdrop-blur-md">
      <div className="flex items-center gap-2 px-3 pt-2 sm:gap-3">
        <button
          onClick={() => { if (!m.playing && m.day >= (t?.days ?? m.steps) - 1) m.setDay(0); m.setPlaying(!m.playing); }}
          disabled={!t}
          className="grid size-8 shrink-0 place-items-center border border-prime/50 bg-prime/10 text-prime transition-all hover:bg-prime/25 disabled:opacity-40"
          aria-label={m.playing ? "Pause" : "Play"}
        >
          {m.playing ? <Pause size={13} /> : <Play size={13} className="ml-0.5" />}
        </button>
        <button
          onClick={() => { m.setPlaying(false); m.setDay(0); }}
          disabled={!t}
          className="grid size-8 shrink-0 place-items-center border border-grid-line bg-abyss text-ink-dim transition-colors hover:text-ink disabled:opacity-40"
          aria-label="Back to day 1"
        >
          <SkipBack size={12} />
        </button>

        <div className="hidden shrink-0 sm:block">
          <Flap text={`DAY ${String(m.day + 1).padStart(2, "0")}`} length={6} />
        </div>

        <div className="min-w-0 flex-1">
          <PixelScrubber
            value={m.day + 1}
            min={1}
            max={t?.days ?? m.steps}
            onChange={(v) => { m.setPlaying(false); m.setDay(v - 1); }}
            labels={(v) => `D${String(v).padStart(2, "0")}`}
          />
        </div>

        {chartOpen && (
          <div className="hidden shrink-0 items-center gap-2 font-mono text-[8px] text-ink-faint md:flex">
            <span className="flex items-center gap-1"><i className="inline-block h-0.5 w-3 bg-prime" /> PRIME</span>
            <span className="flex items-center gap-1"><i className="inline-block h-0.5 w-3 bg-branch" /> BRANCHED</span>
          </div>
        )}
        <button
          onClick={() => setChartOpen((o) => !o)}
          aria-expanded={chartOpen}
          className={`flex h-8 shrink-0 items-center gap-1 border px-2 font-pixel text-[8px] transition-colors ${
            chartOpen ? "border-prime bg-prime/15 text-prime" : "border-grid-line bg-abyss text-ink-dim hover:text-ink"
          }`}
          title="Daily India average, Prime vs Branched"
        >
          <ChartIcon size={11} /> <span className="hidden sm:inline">CHART</span>
          {chartOpen ? <ChevronDown size={10} /> : <ChevronUp size={10} />}
        </button>
      </div>

      {chartOpen && <div className="relative h-[118px] px-1">
        <div className="pointer-events-none absolute left-3 top-1 z-10 font-pixel text-[8px] text-ink-faint">
          {title} <span className="text-ink-dim">({unit})</span>
        </div>
        {t && data.length > 0 ? (
          <LineChart
            data={data}
            style={{ height: "100%" }}
            margin={{ top: 18, right: 26, bottom: 34, left: 52 }}
            animationDuration={700}
            floatingYAxisIds={["left"]}
            revealSignature={`${t.runId}-${m.lens ?? m.variable}`}
          >
            <Grid horizontal numTicksRows={3} />
            <ReferenceArea
              x1={new Date(baseMs + m.day * 86400000 - 9 * 3600000)}
              x2={new Date(baseMs + m.day * 86400000 + 9 * 3600000)}
              fill="#1d6fa3"
              fillOpacity={0.10}
              pattern="none"
            />
            <Line dataKey="Prime" stroke="#1d6fa3" strokeWidth={1.8} curve={curveMonotoneX} fadeEdges={false} />
            <Line dataKey="Branched" stroke="#c07a08" strokeWidth={2.2} curve={curveMonotoneX} glow fadeEdges={false} />
            <YAxis numTicks={3} formatValue={(v) => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(1))} />
            <XAxis numTicks={5} />
            <ChartTooltip showDatePill={false} />
          </LineChart>
        ) : (
          <div className="grid h-full place-items-center font-pixel text-[9px] text-ink-faint mm-pulse">
            AWAITING /simulate
          </div>
        )}
      </div>}
      {!chartOpen && <div className="h-2" />}
    </div>
  );
}
