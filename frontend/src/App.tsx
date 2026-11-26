import { useEffect, useMemo, useState, useCallback } from "react";
import { motion, AnimatePresence } from "motion/react";
import {
  PanelLeft, PanelRight, GitBranch, Eye, Shuffle, AlertTriangle, X,
  Thermometer, CloudRain, Droplets, Wind, Gauge, Globe2, Map as MapIcon,
} from "lucide-react";
import { StatusIsland } from "@/components/kokonutui/status-island";
import PixelGrid from "@/components/PixelGrid";
import GlobeView from "@/components/GlobeView";
import HeaderBar from "@/components/HeaderBar";
import ShockRail from "@/components/ShockRail";
import ImpactRail from "@/components/ImpactRail";
import TimelineDeck from "@/components/TimelineDeck";
import MapLegend from "@/components/MapLegend";
import { PixelTabs } from "@/components/ui";
import { useMultiverse } from "@/state/useMultiverse";
import { computeLens } from "@/lib/impact";
import { VAR_KEYS, VAR_META, type VarKey } from "@/lib/grid";

const EMPTY_MASK = new Map<number, number>();
const VAR_ICONS: Record<VarKey, typeof Wind> = {
  t2m: Thermometer, tp: CloudRain, tcwv: Droplets, u10: Wind, v10: Wind, msl: Gauge,
};

export default function App() {
  const m = useMultiverse();
  const [leftOpen, setLeftOpen] = useState(false);
  const [rightOpen, setRightOpen] = useState(false);
  const [errDismissed, setErrDismissed] = useState(false);
  const [globeMode, setGlobeMode] = useState(true);

  useEffect(() => {
    if (m.error) setErrDismissed(false);
  }, [m.error]);

  const lensOut = useMemo(
    () => (m.timeline && m.lens ? computeLens(m.timeline, m.lens, m.day) : null),
    [m.timeline, m.lens, m.day],
  );

  const showError = m.error && !errDismissed;
  const switchToFlat = useCallback(() => setGlobeMode(false), []);
  const switchToGlobe = useCallback(() => setGlobeMode(true), []);

  return (
    <div className="relative flex h-dvh flex-col overflow-hidden bg-void text-ink">
      <HeaderBar m={m} />

      <StatusIsland
        tasks={m.tasks}
        onDismiss={m.dismissTask}
        dismissAfter={4200}
        className="absolute left-1/2 top-16 z-50 -translate-x-1/2"
      />

      <div className="relative z-10 flex min-h-0 flex-1">
        {/* ── left rail: shock painter ── */}
        <aside className="hidden w-[288px] shrink-0 border-r border-grid-line bg-abyss/70 backdrop-blur-md lg:block">
          <ShockRail m={m} />
        </aside>
        <AnimatePresence>
          {leftOpen && (
            <>
              <motion.div
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                className="absolute inset-0 z-40 bg-void/70 backdrop-blur-sm lg:hidden"
                onClick={() => setLeftOpen(false)}
              />
              <motion.aside
                initial={{ x: -300 }} animate={{ x: 0 }} exit={{ x: -300 }}
                transition={{ type: "spring", stiffness: 420, damping: 38 }}
                className="absolute inset-y-0 left-0 z-50 w-[288px] border-r border-grid-line bg-abyss/95 backdrop-blur-md lg:hidden"
              >
                <button
                  onClick={() => setLeftOpen(false)}
                  aria-label="Close shock painter"
                  className="absolute right-2 top-2 z-10 grid size-7 place-items-center border border-grid-line bg-abyss text-ink-dim hover:text-ink"
                >
                  <X size={13} />
                </button>
                <div className="h-full pt-9">
                  <ShockRail m={m} />
                </div>
              </motion.aside>
            </>
          )}
        </AnimatePresence>

        {/* ── centre: globe or flat map + timeline ── */}
        <main className="flex min-w-0 flex-1 flex-col">
          <div className="relative min-h-0 flex-1">
            {/* view + variable overlays */}
            <div className="pointer-events-none absolute inset-x-2 top-2 z-20 flex flex-wrap items-start justify-between gap-2">
              <div className="pointer-events-auto flex flex-wrap items-center gap-1.5">
                <div className="flex gap-0.5 border border-grid-line bg-abyss/80 p-0.5">
                  <button
                    onClick={switchToGlobe}
                    className={`flex items-center gap-1 px-1.5 py-1 font-pixel text-[8px] transition-colors ${
                      globeMode ? "bg-safe text-white" : "text-ink-dim hover:text-ink"
                    }`}
                    title="Interactive pixel-art Radio Garden globe"
                  >
                    <Globe2 size={10} />
                    <span>GLOBE</span>
                  </button>
                  <button
                    onClick={switchToFlat}
                    className={`flex items-center gap-1 px-1.5 py-1 font-pixel text-[8px] transition-colors ${
                      !globeMode ? "bg-prime text-white" : "text-ink-dim hover:text-ink"
                    }`}
                    title="2D flat grid map"
                  >
                    <MapIcon size={10} />
                    <span className="hidden sm:inline">FLAT</span>
                  </button>
                </div>

                <PixelTabs
                  value={m.view}
                  onChange={(v) => m.setView(v as typeof m.view)}
                  items={[
                    { id: "prime", label: "PRIME", icon: Eye, accent: "#1d6fa3" },
                    { id: "branched", label: "BRANCHED", icon: GitBranch, accent: "#c07a08" },
                    { id: "diff", label: "DIFF Δ", icon: Shuffle, accent: "#c53d2e" },
                  ]}
                />
              </div>

              <div className="pointer-events-auto flex gap-0.5 border border-grid-line bg-abyss/80 p-0.5">
                {m.lens ? (
                  <span className="px-2 py-1 font-pixel text-[8px] text-ink-faint">
                    LENS ACTIVE — RAW FIELDS OFF
                  </span>
                ) : (
                  VAR_KEYS.map((v) => {
                    const Icon = VAR_ICONS[v];
                    const active = m.variable === v;
                    return (
                      <button
                        key={v}
                        onClick={() => m.setVariable(v)}
                        title={VAR_META[v].label}
                        className={`flex items-center gap-1 px-1.5 py-1 font-pixel text-[8px] transition-colors ${
                          active ? "bg-prime text-void" : "text-ink-dim hover:text-ink"
                        }`}
                      >
                        <Icon size={9} strokeWidth={2.5} />
                        <span className="hidden sm:inline">{VAR_META[v].short}</span>
                      </button>
                    );
                  })
                )}
              </div>
            </div>

            <AnimatePresence mode="wait">
              {globeMode ? (
                <motion.div
                  key="globe"
                  initial={{ opacity: 0, scale: 0.96 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 1.04 }}
                  transition={{ duration: 0.3, ease: "easeOut" }}
                  className="absolute inset-0"
                >
                  <GlobeView
                    timeline={m.timeline}
                    landSeaMask={m.landSeaMask}
                    view={m.view}
                    variable={m.variable}
                    day={m.day}
                    lens={m.lens}
                    lensOut={lensOut}
                    paintMask={m.painted?.mask ?? EMPTY_MASK}
                    paintMode={m.paintMode}
                    brushRadius={m.brushRadius}
                    onPaint={m.onPaint}
                    onPaintEnd={m.onPaintEnd}
                    selected={m.selected}
                    onSelect={m.selectCell}
                    ripples={m.ripples}
                    showWind={m.showWind}
                    showCities={m.showCities}
                    showScenery={m.showScenery}
                    weatherFocus={m.weatherFocus}
                    running={m.running}
                    focusRegion={m.focusRegion}
                    onSwitchToFlatMap={switchToFlat}
                  />
                </motion.div>
              ) : (
                <motion.div
                  key="flat-map"
                  initial={{ opacity: 0, scale: 1.03 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.97 }}
                  transition={{ duration: 0.3, ease: "easeOut" }}
                  className="absolute inset-0"
                >
                  <PixelGrid
                    timeline={m.timeline}
                    landSeaMask={m.landSeaMask}
                    view={m.view}
                    variable={m.variable}
                    day={m.day}
                    lens={m.lens}
                    lensOut={lensOut}
                    paintMask={m.painted?.mask ?? EMPTY_MASK}
                    paintMode={m.paintMode}
                    brushRadius={m.brushRadius}
                    onPaint={m.onPaint}
                    onPaintEnd={m.onPaintEnd}
                    selected={m.selected}
                    onSelect={m.selectCell}
                    ripples={m.ripples}
                    showWind={m.showWind}
                    showCities={m.showCities}
                    showScenery={m.showScenery}
                    weatherFocus={m.weatherFocus}
                    running={m.running}
                    focusRegion={m.focusRegion}
                  />
                </motion.div>
              )}
            </AnimatePresence>

            <MapLegend m={m} lensOut={lensOut} />

            {/* first-run guidance */}
            <AnimatePresence>
              {m.booted && !m.diverged && m.perturbationCount === 0 && !m.paintMode && (
                <motion.div
                  initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                  className="pointer-events-none absolute bottom-14 left-1/2 z-10 hidden -translate-x-1/2 panel px-3 py-1.5 font-pixel text-[8px] text-ink-dim md:bottom-2 md:block"
                >
                  {globeMode && (
                    <>
                      <span className="text-safe">SPIN EARTH</span>
                      <span className="mx-1.5 text-ink-faint">·</span>
                    </>
                  )}
                  <span className="text-shock">1 PAINT A SHOCK</span>
                  <span className="mx-1.5 text-ink-faint">→</span>
                  <span className="text-branch">2 BRANCH</span>
                  <span className="mx-1.5 text-ink-faint">→</span>
                  <span className="text-prime">3 WATCH IT RIPPLE</span>
                </motion.div>
              )}
            </AnimatePresence>

            {/* error banner */}
            <AnimatePresence>
              {showError && (
                <motion.div
                  initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 12 }}
                  className="absolute bottom-12 left-1/2 z-30 flex w-[min(480px,90%)] -translate-x-1/2 items-center gap-2 border border-danger/60 bg-danger/15 px-3 py-2 backdrop-blur-md"
                >
                  <AlertTriangle size={14} className="shrink-0 text-danger" />
                  <span className="min-w-0 flex-1 font-mono text-[10px] text-ink">{m.error}</span>
                  <button onClick={() => setErrDismissed(true)} className="text-ink-faint hover:text-ink" aria-label="Dismiss error">
                    <X size={12} />
                  </button>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          <TimelineDeck m={m} />
        </main>

        {/* ── right rail: impact + inspector ── */}
        <aside className="hidden w-[324px] shrink-0 border-l border-grid-line bg-abyss/70 backdrop-blur-md xl:block">
          <ImpactRail m={m} />
        </aside>
        <AnimatePresence>
          {rightOpen && (
            <>
              <motion.div
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                className="absolute inset-0 z-40 bg-void/70 backdrop-blur-sm xl:hidden"
                onClick={() => setRightOpen(false)}
              />
              <motion.aside
                initial={{ x: 340 }} animate={{ x: 0 }} exit={{ x: 340 }}
                transition={{ type: "spring", stiffness: 420, damping: 38 }}
                className="absolute inset-y-0 right-0 z-50 w-[324px] border-l border-grid-line bg-abyss/95 backdrop-blur-md xl:hidden"
              >
                <button
                  onClick={() => setRightOpen(false)}
                  aria-label="Close impact panel"
                  className="absolute right-2 top-2 z-10 grid size-7 place-items-center border border-grid-line bg-abyss text-ink-dim hover:text-ink"
                >
                  <X size={13} />
                </button>
                <div className="h-full pt-9">
                  <ImpactRail m={m} />
                </div>
              </motion.aside>
            </>
          )}
        </AnimatePresence>
      </div>

      {/* mobile rail toggles — lifted cleanly above MapLegend & Tuned Station card */}
      {!leftOpen && (
        <div className="pointer-events-none absolute bottom-[284px] left-2 z-30 lg:hidden">
          <button
            onClick={() => { setLeftOpen(true); setRightOpen(false); }}
            className="pointer-events-auto grid size-9 place-items-center border border-shock/50 bg-abyss/90 text-shock backdrop-blur-md"
            aria-label="Open shock painter"
          >
            <PanelLeft size={15} />
          </button>
        </div>
      )}
      {!rightOpen && (
        <div className="pointer-events-none absolute bottom-[284px] right-2 z-30 xl:hidden">
          <button
            onClick={() => { setRightOpen(true); setLeftOpen(false); }}
            className="pointer-events-auto grid size-9 place-items-center border border-branch/50 bg-abyss/90 text-branch backdrop-blur-md"
            aria-label="Open impact panel"
          >
            <PanelRight size={15} />
          </button>
        </div>
      )}
    </div>
  );
}
