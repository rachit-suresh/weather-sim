import { useMemo } from "react";
import { motion, AnimatePresence } from "motion/react";
import {
  Paintbrush, Eraser, Thermometer, CloudRain, Droplets, Wind, Gauge,
  GitBranch, Trash2, Waves, MapPin, Crosshair, Sparkles,
} from "lucide-react";
import LiquidSlider from "@/components/kokonutui/liquid-slider";
import DetentDial from "@/components/kokonutui/detent-dial";
import HoldButton from "@/components/kokonutui/hold-button";
import { SectionTitle, ToggleChip } from "@/components/ui";
import { VAR_META, VAR_KEYS, GRID, displayUnit, type VarKey } from "@/lib/grid";
import { JUDGE_PRESETS, MAX_PERTURBATIONS, PRESETS, type Multiverse } from "@/state/useMultiverse";

const VAR_ICONS: Record<VarKey, typeof Wind> = {
  t2m: Thermometer, tp: CloudRain, tcwv: Droplets, u10: Wind, v10: Wind, msl: Gauge,
};

const DELTA_RANGE: Record<VarKey, { min: number; max: number; step: number }> = {
  t2m: { min: -5, max: 5, step: 0.5 },
  tp: { min: -50, max: 50, step: 5 },
  tcwv: { min: -20, max: 20, step: 1 },
  u10: { min: -8, max: 8, step: 0.5 },
  v10: { min: -8, max: 8, step: 0.5 },
  msl: { min: -8, max: 8, step: 0.5 },
};

export default function ShockRail({ m }: { m: Multiverse }) {
  const isPct = m.paintValueMode === "pct";
  const range = isPct ? { min: -50, max: 50, step: 5 } : DELTA_RANGE[m.paintField];
  const unit = isPct ? "%" : displayUnit(m.paintField);

  const paintedCells = m.painted?.mask.size ?? 0;
  const shockSummary = useMemo(() => {
    const parts: string[] = [];
    if (m.painted) parts.push(`${VAR_META[m.painted.field].short} ${m.painted.value > 0 ? "+" : ""}${m.painted.value}`);
    for (const q of m.queue) parts.push(q.label);
    return parts;
  }, [m.painted, m.queue]);

  return (
    <div className="flex h-full flex-col overflow-y-auto">
      {/* ── painter ── */}
      <SectionTitle
        right={
          <ToggleChip active={m.paintMode} onClick={() => m.setPaintMode(!m.paintMode)} icon={Paintbrush} accent="var(--color-shock)">
            {m.paintMode ? "PAINTING" : "PAINT"}
          </ToggleChip>
        }
      >
        SHOCK PAINTER
      </SectionTitle>

      <div className="px-3 pb-2">
        <div className="flex items-start justify-between gap-2">
          <p className="text-[10px] leading-snug text-ink-faint">
            {m.paintMode
              ? "Draw on the map to perturb the initial condition. Shift-drag erases."
              : "Arm the brush, then draw a localized shock anywhere on the grid."}
          </p>
          <div className="inline-flex shrink-0 border border-grid-line bg-abyss p-0.5 font-pixel text-[7px]">
            <button
              type="button"
              onClick={() => m.setValueMode("abs")}
              className={`px-1 py-0.5 ${!isPct ? "bg-prime text-void" : "text-ink-dim"}`}
              title="Absolute physical-unit delta"
            >
              ABS
            </button>
            <button
              type="button"
              onClick={() => m.setValueMode("pct")}
              className={`px-1 py-0.5 ${isPct ? "bg-prime text-void" : "text-ink-dim"}`}
              title="Percentage of latest input day"
            >
              %
            </button>
          </div>
        </div>

        <div className="mt-2.5 grid grid-cols-3 gap-1">
          {VAR_KEYS.map((v) => {
            const Icon = VAR_ICONS[v];
            const active = m.paintField === v;
            return (
              <button
                key={v}
                onClick={() => m.setField(v)}
                className={`flex flex-col items-center gap-0.5 border py-1.5 font-pixel text-[8px] transition-all ${
                  active
                    ? "border-shock/70 bg-shock/15 text-shock glow-shock"
                    : "border-grid-line bg-abyss/60 text-ink-dim hover:border-shock/40 hover:text-ink"
                }`}
                title={VAR_META[v].label}
              >
                <Icon size={12} />
                {VAR_META[v].short}
              </button>
            );
          })}
        </div>

        <div className="mt-3 flex gap-2">
          <div className="relative shrink-0 overflow-hidden" style={{ width: 174, height: 224 }}>
            <div className="origin-top-left" style={{ transform: "scale(0.62)" }}>
              <LiquidSlider
                label={`Δ ${VAR_META[m.paintField].label}`}
                value={m.paintValue}
                onValueChange={m.setDeltaValue}
                min={range.min}
                max={range.max}
                step={range.step}
                unit={unit}
                formatValue={(v) => `${v > 0 ? "+" : ""}${v}`}
                className="border-grid-line bg-abyss"
              />
            </div>
          </div>
          <div className="flex min-w-0 flex-1 flex-col items-center gap-2">
            <div className="relative shrink-0 overflow-hidden" style={{ width: 86, height: 92 }}>
              <div className="origin-top-left" style={{ transform: "scale(0.66)" }}>
                <DetentDial
                  label="BRUSH"
                  min={1}
                  max={5}
                  step={0.5}
                  value={m.brushRadius}
                  onValueChange={m.setBrushRadius}
                  size={130}
                  formatValue={(v) => v.toFixed(1)}
                />
              </div>
            </div>
            <div className="text-center">
              <div className="font-pixel text-[8px] text-ink-faint">CELLS</div>
              <div className="font-mono text-xl text-shock tabular-nums">{paintedCells}</div>
            </div>
            <button onClick={m.clearPaint} disabled={!m.painted} aria-label="Clear painted shock" className="flex items-center gap-1 text-[10px] text-ink-dim hover:text-danger disabled:opacity-35">
              <Eraser size={12} /> Clear
            </button>
          </div>
        </div>
      </div>

      {/* ── armed shocks + run ── */}
      <SectionTitle right={<span className="font-mono text-[9px] text-ink-faint">{m.perturbationCount}/{MAX_PERTURBATIONS}</span>}>
        ARMED SHOCKS
      </SectionTitle>
      <div className="px-3 pb-2">
        <AnimatePresence initial={false}>
          {shockSummary.length === 0 && (
            <motion.div
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
              className="border border-dashed border-grid-line px-2 py-2 text-center text-[9px] text-ink-faint"
            >
              nothing armed — paint or pick a preset
            </motion.div>
          )}
          {m.painted && (
            <motion.div
              layout initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: 12 }}
              className="mb-1 flex items-center justify-between gap-2 border border-shock/50 bg-shock/10 px-2 py-1"
            >
              <span className="flex items-center gap-1.5 font-mono text-[10px] text-shock">
                <Paintbrush size={10} />
                {VAR_META[m.painted.field].short} {m.painted.value > 0 ? "+" : ""}{m.painted.value} {m.painted.valueMode === "pct" ? "%" : displayUnit(m.painted.field)} · {paintedCells}c
              </span>
              <button onClick={m.clearPaint} className="text-ink-faint hover:text-danger" aria-label="Remove painted shock">
                <Trash2 size={11} />
              </button>
            </motion.div>
          )}
          {m.queue.map((q) => (
            <motion.div
              key={q.id} layout
              initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: 12 }}
              className="mb-1 flex items-center justify-between gap-2 border border-grid-line bg-abyss/60 px-2 py-1"
            >
              <span className="flex items-center gap-1.5 font-mono text-[10px] text-ink-dim">
                <Crosshair size={10} className="text-branch" /> {q.label}
              </span>
              <button onClick={() => m.removeShock(q.id)} className="text-ink-faint hover:text-danger" aria-label={`Remove ${q.label}`}>
                <Trash2 size={11} />
              </button>
            </motion.div>
          ))}
        </AnimatePresence>

        <motion.button
          whileTap={{ scale: 0.97 }}
          onClick={() => void m.run()}
          disabled={m.perturbationCount === 0 || m.running}
          className={`mt-2 flex w-full items-center justify-center gap-2 border py-2.5 font-pixel text-[10px] transition-all ${
            m.perturbationCount > 0 && !m.running
              ? "border-shock bg-shock/20 text-shock glow-shock hover:bg-shock/30"
              : "border-grid-line bg-abyss/60 text-ink-faint opacity-60"
          }`}
        >
          <GitBranch size={13} className={m.running ? "mm-pulse" : ""} />
          {m.running ? "SIMULATING…" : "BRANCH TIMELINE"}
        </motion.button>

        <HoldButton
          variant="red"
          holdDuration={900}
          holdingLabel="Keep holding…"
          completeLabel="Re-converged"
          onHoldComplete={m.reset}
          className="mt-1.5 w-full font-pixel text-[8px]"
        >
          HOLD TO RESET PRIME
        </HoldButton>
      </div>

      {/* ── official judge shocks ── */}
      <SectionTitle>JUDGE SHOCKS</SectionTitle>
      <div className="space-y-1 px-3 pb-2">
        {JUDGE_PRESETS.map((p) => {
          const armed = m.queue.some((q) => q.group === p.id);
          return (
            <button
              key={p.id}
              onClick={() => (armed ? m.removeJudgePreset(p.id) : m.addJudgePreset(p))}
              className={`group w-full border px-2 py-1.5 text-left transition-all ${
                armed
                  ? "border-shock/60 bg-shock/10 glow-shock"
                  : "border-grid-line bg-abyss/60 hover:border-shock/40"
              }`}
              title={p.blurb}
            >
              <div className="flex items-center justify-between gap-2">
                <span className={`truncate font-pixel text-[8px] ${armed ? "text-shock" : "text-ink-dim group-hover:text-ink"}`}>
                  {p.label}
                </span>
                <span className={`shrink-0 font-mono text-[9px] ${armed ? "text-shock" : "text-ink-faint"}`}>
                  {armed ? "ARMED ✓" : "ARM"}
                </span>
              </div>
              <div className="mt-0.5 truncate font-mono text-[8px] text-ink-faint">{p.blurb}</div>
            </button>
          );
        })}
      </div>

      {/* ── presets ── */}
      <SectionTitle>REGION PRESETS</SectionTitle>
      <div className="space-y-1 px-3 pb-2">
        {PRESETS.map((p) => {
          const armed = m.queue.some((q) => q.label === p.label);
          return (
            <button
              key={p.label}
              onClick={() => (armed ? m.removeShock(m.queue.find((q) => q.label === p.label)!.id) : m.addPreset(p))}
              className={`group w-full border px-2 py-1.5 text-left transition-all ${
                armed
                  ? "border-shock/60 bg-shock/10 glow-shock"
                  : "border-grid-line bg-abyss/60 hover:border-shock/40"
              }`}
              title={p.blurb}
            >
              <div className="flex items-center justify-between gap-2">
                <span className={`truncate font-pixel text-[8px] ${armed ? "text-shock" : "text-ink-dim group-hover:text-ink"}`}>
                  {p.label}
                </span>
                <span className={`shrink-0 font-mono text-[9px] ${armed ? "text-shock" : "text-ink-faint"}`}>
                  {armed ? "ARMED ✓" : "ARM"}
                </span>
              </div>
            </button>
          );
        })}
      </div>

      {/* ── display ── */}
      <SectionTitle>DISPLAY</SectionTitle>
      <div className="px-3 pb-4">
        <div className="flex flex-wrap gap-1">
          <ToggleChip active={m.showWind} onClick={() => m.setShowWind(!m.showWind)} icon={Wind}>WIND</ToggleChip>
          <ToggleChip active={m.showCities} onClick={() => m.setShowCities(!m.showCities)} icon={MapPin}>CITIES</ToggleChip>
          {/* spec IMP-3d: on = Storybook (pixel-art map + forecast tint), off = Data (full colour field) */}
          <span title={m.showScenery ? "Storybook: pixel-art map with the forecast as a tint. Click for Data mode (full colour field)." : "Data mode: full colour field. Click for the Storybook pixel-art map."}>
            <ToggleChip active={m.showScenery} onClick={() => m.setShowScenery(!m.showScenery)} icon={Sparkles} accent="var(--color-safe)">STORYBOOK</ToggleChip>
          </span>
        </div>
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
        <div className="mt-2 flex items-center gap-1.5 text-[9px] text-ink-faint">
          <Waves size={10} className="text-prime" />
          Shocks evolve freely after day 0 — watch the ripple.
        </div>
      </div>
    </div>
  );
}
