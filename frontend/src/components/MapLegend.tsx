import { useMemo } from "react";
import {
  ABS_RANGE, SEQUENTIAL, divergingColor, moistColor, HEAT_RAMP, ENERGY_RAMP,
  diffScale, DIFF_FLOOR, rgbCss,
} from "@/lib/colormaps";
import { diffFrame } from "@/lib/api";
import { CELLS, isIndia, rowOf, colOf, display, displayDelta, displayUnit } from "@/lib/grid";
import { lensById, type LensOutput } from "@/lib/impact";
import type { Multiverse } from "@/state/useMultiverse";

export default function MapLegend({ m, lensOut }: { m: Multiverse; lensOut: LensOutput | null }) {
  const t = m.timeline;

  const spec = useMemo(() => {
    if (!t) return null;
    const stops = 13;
    const sample = (fn: (x: number) => [number, number, number]) =>
      Array.from({ length: stops }, (_, k) => rgbCss(fn(k / (stops - 1)))).join(", ");

    if (m.lens && lensOut) {
      const def = lensById(m.lens);
      if (m.view === "diff") {
        const s = lensOut.scale;
        const fn = m.lens === "flood" ? moistColor : m.lens === "energy" ? (x: number) => ENERGY_RAMP((x + 1) / 2) : divergingColor;
        return {
          gradient: sample((x) => fn(x * 2 - 1)),
          labels: [-s, 0, s].map((v) => `${v > 0 ? "+" : ""}${v.toFixed(Math.abs(s) < 1 ? 2 : 1)}`),
          title: `Δ ${def.short} (${def.unit})`,
        };
      }
      if (m.lens === "heat") return { gradient: sample(HEAT_RAMP), labels: ["22", "28.5", "35"], title: "WET-BULB °C" };
      if (m.lens === "flood") return { gradient: sample(SEQUENTIAL.tp), labels: ["0", "50", "100"], title: "FLOOD RISK INDEX" };
      return { gradient: sample(ENERGY_RAMP), labels: ["−6", "+2", "+10"], title: "WATER BALANCE mm/d" };
    }

    if (m.view === "diff") {
      const d = diffFrame(t, m.variable, m.day);
      const mask = new Array<number>(CELLS);
      for (let i = 0; i < CELLS; i++) mask[i] = isIndia(rowOf(i), colOf(i)) ? 1 : 0;
      const s = displayDelta(m.variable, diffScale(d, DIFF_FLOOR, m.variable, mask));
      const u = displayUnit(m.variable);
      return {
        gradient: sample((x) => divergingColor(x * 2 - 1)),
        labels: [-s, 0, s].map((v) => `${v > 0 ? "+" : ""}${v.toFixed(Math.abs(s) < 1 ? 2 : 1)}`),
        title: `Δ ${m.variable.toUpperCase()} (${u})`,
      };
    }

    const [lo, hi] = ABS_RANGE[m.variable];
    const dLo = display(m.variable, lo);
    const dHi = display(m.variable, hi);
    return {
      gradient: sample(SEQUENTIAL[m.variable]),
      labels: [dLo, (dLo + dHi) / 2, dHi].map((v) => v.toFixed(Math.abs(dHi - dLo) < 20 ? 1 : 0)),
      title: `${m.variable.toUpperCase()} (${displayUnit(m.variable)})`,
    };
  }, [t, m.lens, lensOut, m.view, m.variable, m.day]);

  if (!spec) return null;

  return (
    <div className="pointer-events-none absolute bottom-2 left-2 z-10 w-[168px] panel px-2 py-1.5">
      <div className="font-pixel text-[7px] text-ink-faint">{spec.title}{m.showScenery && <span className="ml-1 text-[6px]">· TINT</span>}</div>
      <div className="mt-1 h-2 w-full border border-grid-line" style={{ background: `linear-gradient(90deg, ${spec.gradient})` }} />
      <div className="mt-0.5 flex justify-between font-mono text-[8px] text-ink-dim tabular-nums">
        {spec.labels.map((l, k) => <span key={k}>{l}</span>)}
      </div>
    </div>
  );
}
