import { useEffect, useRef, useState } from "react";
import { SlidersHorizontal, Wind, MapPin, Sparkles, ChevronDown } from "lucide-react";
import { ToggleChip } from "@/components/ui";
import { GRID } from "@/lib/grid";
import type { Multiverse } from "@/state/useMultiverse";

/** spec IMP-4c: display settings in one dropdown instead of a panel section. */
export default function ViewMenu({ m }: { m: Multiverse }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent) => { if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false); };
    const esc = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    document.addEventListener("mousedown", close);
    document.addEventListener("keydown", esc);
    return () => { document.removeEventListener("mousedown", close); document.removeEventListener("keydown", esc); };
  }, [open]);

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        aria-label="View settings"
        className={`flex h-[26px] items-center gap-1 border px-2 font-pixel text-[8px] transition-colors ${
          open ? "border-prime bg-prime text-white" : "border-grid-line bg-abyss/80 text-ink-dim hover:text-ink"
        }`}
      >
        <SlidersHorizontal size={10} /> VIEW <ChevronDown size={9} />
      </button>
      {open && (
        <div className="panel absolute right-0 top-full z-40 mt-1.5 w-[236px] p-2.5">
          <div className="flex flex-wrap gap-1">
            <ToggleChip active={m.showWind} onClick={() => m.setShowWind(!m.showWind)} icon={Wind}>WIND</ToggleChip>
            <ToggleChip active={m.showCities} onClick={() => m.setShowCities(!m.showCities)} icon={MapPin}>CITIES</ToggleChip>
            <span title={m.showScenery ? "Storybook: pixel-art map with the forecast as a tint. Click for Data mode (full colour field)." : "Data mode: full colour field. Click for the Storybook pixel-art map."}>
              <ToggleChip active={m.showScenery} onClick={() => m.setShowScenery(!m.showScenery)} icon={Sparkles} accent="var(--color-safe)">STORYBOOK</ToggleChip>
            </span>
          </div>
          <label
            className={`mt-2.5 block ${m.showScenery ? "" : "opacity-40"}`}
            title="Fade the map art and show the forecast colours at full strength. Works in Storybook mode."
          >
            <span className="flex items-center justify-between font-pixel text-[8px] text-ink-faint">
              <span>WEATHER FOCUS</span>
              <span className="font-mono text-[10px] text-ink">{Math.round(m.weatherFocus * 100)}%</span>
            </span>
            <input
              type="range" min={0} max={1} step={0.05}
              value={m.weatherFocus}
              disabled={!m.showScenery}
              onChange={(e) => m.setWeatherFocus(Number(e.target.value))}
              aria-label="Weather focus"
              className="mt-1 w-full cursor-pointer accent-[var(--color-prime)] disabled:cursor-not-allowed"
            />
          </label>
          <label className="mt-2 block">
            <span className="font-pixel text-[8px] text-ink-faint">FOCUS REGION</span>
            <select
              value={m.focusRegion ?? ""}
              onChange={(e) => m.setFocusRegion(e.target.value || null)}
              className="mt-1 w-full border border-grid-line bg-abyss px-2 py-1.5 font-mono text-[10px] text-ink outline-none focus:border-prime/60"
            >
              <option value="">All India</option>
              {GRID.regions.map((r) => (
                <option key={r.name} value={r.name}>{r.name}</option>
              ))}
            </select>
          </label>
        </div>
      )}
    </div>
  );
}
