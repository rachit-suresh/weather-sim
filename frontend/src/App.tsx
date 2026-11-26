import { useEffect, useMemo, useState, useCallback } from "react";
import { motion, AnimatePresence } from "motion/react";
import {
  GitBranch, Eye, Shuffle, AlertTriangle, X, Zap, BarChart3, Paintbrush,
  Globe2, Map as MapIcon, Layers,
} from "lucide-react";
import { StatusIsland } from "@/components/kokonutui/status-island";
import PixelGrid from "@/components/PixelGrid";
import GlobeView from "@/components/GlobeView";
import HeaderBar from "@/components/HeaderBar";
import ShockRail from "@/components/ShockRail";
import ImpactRail from "@/components/ImpactRail";
import TimelineDeck from "@/components/TimelineDeck";
import MapLegend from "@/components/MapLegend";
import ViewMenu from "@/components/ViewMenu";
import HelpModal from "@/components/HelpModal";
import { PixelTabs } from "@/components/ui";
import { useMultiverse } from "@/state/useMultiverse";
import { computeLens, lensById } from "@/lib/impact";
import { VAR_KEYS, VAR_META, type VarKey } from "@/lib/grid";

const EMPTY_MASK = new Map<number, number>();

/** spec IMP-4c: plain names for the six forecast variables (codes kept in brackets). */
const VAR_PLAIN: Record<VarKey, string> = {
  tp: "Rainfall", t2m: "Temperature", u10: "Wind east–west", v10: "Wind north–south", msl: "Pressure", tcwv: "Moisture",
};

/* spec IMP-4: on load only the header, the map and the day slider show; everything else opens on demand. */
export default function App() {
  const m = useMultiverse();
  const [leftOpen, setLeftOpen] = useState(false);
  const [rightOpen, setRightOpen] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);
  const [impactNew, setImpactNew] = useState(false);
  const [errDismissed, setErrDismissed] = useState(false);
  const [globeMode, setGlobeMode] = useState(true);

  useEffect(() => {
    if (m.error) setErrDismissed(false);
  }, [m.error]);

  // a fresh branch result: mark the Impact button (IMP-4b)
  useEffect(() => {
    if (m.diverged && m.timeline) setImpactNew(true);
  }, [m.timeline?.runId]); // eslint-disable-line react-hooks/exhaustive-deps

  const openLeft = useCallback(() => { setLeftOpen(true); setRightOpen(false); }, []);
  const openRight = useCallback(() => { setRightOpen(true); setLeftOpen(false); setImpactNew(false); }, []);

  // clicking a cell opens its inspector in the Impact drawer (IMP-4b)
  useEffect(() => {
    if (m.selected !== null) openRight();
  }, [m.selected, openRight]);

  // painting needs the map: close the drawer while the brush is armed
  useEffect(() => {
    if (m.paintMode) setLeftOpen(false);
  }, [m.paintMode]);

  const lensOut = useMemo(
    () => (m.timeline && m.lens ? computeLens(m.timeline, m.lens, m.day) : null),
    [m.timeline, m.lens, m.day],
  );

  const showError = m.error && !errDismissed;
  const switchToFlat = useCallback(() => setGlobeMode(false), []);
  const switchToGlobe = useCallback(() => setGlobeMode(true), []);
  const queued = m.perturbationCount;

  const mapProps = {
    timeline: m.timeline,
    landSeaMask: m.landSeaMask,
    view: m.view,
    variable: m.variable,
    day: m.day,
    lens: m.lens,
    lensOut,
    paintMask: m.painted?.mask ?? EMPTY_MASK,
    paintMode: m.paintMode,
    brushRadius: m.brushRadius,
    onPaint: m.onPaint,
    onPaintEnd: m.onPaintEnd,
    selected: m.selected,
    onSelect: m.selectCell,
    ripples: m.ripples,
    showWind: m.showWind,
    showCities: m.showCities,
    showScenery: m.showScenery,
    weatherFocus: m.weatherFocus,
    running: m.running,
    focusRegion: m.focusRegion,
  };

  return (
    <div className="relative flex h-dvh flex-col overflow-hidden bg-void text-ink">
      <HeaderBar m={m} onHelp={() => setHelpOpen(true)} />

      <StatusIsland
        tasks={m.tasks}
        onDismiss={m.dismissTask}
        dismissAfter={4200}
        className="absolute left-1/2 top-16 z-50 -translate-x-1/2"
      />

      <div className="relative z-10 flex min-h-0 flex-1">
        {/* ── left drawer: shocks (IMP-4a) ── */}
        <Drawer side="left" open={leftOpen} onClose={() => setLeftOpen(false)} label="Close shocks panel" width={300}>
          <ShockRail m={m} />
        </Drawer>

        {/* ── centre: globe or flat map + timeline ── */}
        <main className="flex min-w-0 flex-1 flex-col">
          <div className="relative min-h-0 flex-1">
            {/* toolbar */}
            <div className="pointer-events-none absolute inset-x-2 top-2 z-20 flex flex-wrap items-start justify-between gap-2">
              <div className="pointer-events-auto flex flex-wrap items-center gap-1.5">
                <button
                  onClick={openLeft}
                  className="relative flex h-[26px] items-center gap-1.5 border border-shock bg-shock px-2.5 font-pixel text-[8px] text-white shadow-[2px_2px_0_rgba(51,32,15,0.18)] hover:brightness-110"
                  aria-label="Open shocks panel"
                >
                  <Zap size={11} /> SHOCKS
                  {queued > 0 && (
                    <span className="ml-0.5 grid min-w-4 place-items-center bg-white px-1 font-mono text-[9px] text-shock">{queued}</span>
                  )}
                </button>

                <div className="flex gap-0.5 border border-grid-line bg-abyss/80 p-0.5">
                  <button
                    onClick={switchToGlobe}
                    className={`flex items-center gap-1 px-1.5 py-1 font-pixel text-[8px] transition-colors ${
                      globeMode ? "bg-safe text-white" : "text-ink-dim hover:text-ink"
                    }`}
                    title="Globe view"
                  >
                    <Globe2 size={10} />
                    <span className="hidden sm:inline">GLOBE</span>
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

              <div className="pointer-events-auto flex flex-wrap items-center justify-end gap-1.5">
                {m.lens ? (
                  <button
                    onClick={() => m.setLens(null)}
                    className="flex h-[26px] items-center gap-1 border border-branch/60 bg-branch/15 px-2 font-pixel text-[8px] text-branch hover:bg-branch/25"
                    title="An impact lens is showing; click to go back to the raw forecast variables"
                  >
                    <Layers size={10} /> {lensById(m.lens).short} LENS <X size={10} />
                  </button>
                ) : (
                  <select
                    value={m.variable}
                    onChange={(e) => m.setVariable(e.target.value as VarKey)}
                    aria-label="Forecast variable"
                    title="Which forecast variable the map shows"
                    className="h-[26px] border border-grid-line bg-abyss/90 px-1.5 font-mono text-[10px] text-ink outline-none focus:border-prime/60"
                  >
                    {VAR_KEYS.map((v) => (
                      <option key={v} value={v}>{VAR_PLAIN[v]} ({VAR_META[v].short})</option>
                    ))}
                  </select>
                )}
                <ViewMenu m={m} />
                <button
                  onClick={openRight}
                  className="relative flex h-[26px] items-center gap-1.5 border border-branch bg-branch px-2.5 font-pixel text-[8px] text-white shadow-[2px_2px_0_rgba(51,32,15,0.18)] hover:brightness-110"
                  aria-label="Open impact panel"
                >
                  <BarChart3 size={11} /> IMPACT
                  {impactNew && <span className="absolute -right-1 -top-1 size-2.5 border border-white bg-danger mm-pulse" aria-label="new result" />}
                </button>
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
                  <GlobeView {...mapProps} onSwitchToFlatMap={switchToFlat} />
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
                  <PixelGrid {...mapProps} />
                </motion.div>
              )}
            </AnimatePresence>

            <MapLegend m={m} lensOut={lensOut} />

            {/* bottom-centre actions: first-run hint, painting controls, branch button */}
            <div className="pointer-events-none absolute bottom-2 left-1/2 z-30 flex -translate-x-1/2 flex-wrap items-center justify-center gap-1.5">
              <AnimatePresence>
                {m.booted && !m.diverged && queued === 0 && !m.paintMode && !leftOpen && (
                  <motion.button
                    key="hint"
                    initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                    onClick={openLeft}
                    className="pointer-events-auto panel px-3 py-1.5 font-pixel text-[8px] text-ink-dim hover:text-ink"
                  >
                    START: <span className="text-shock">SHOCKS</span> → PICK ONE → <span className="text-branch">BRANCH</span>
                  </motion.button>
                )}
                {m.paintMode && (
                  <motion.div
                    key="paint"
                    initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                    className="pointer-events-auto flex items-center gap-2 border border-shock/60 bg-panel/95 px-2.5 py-1 font-pixel text-[8px] text-shock"
                  >
                    <Paintbrush size={11} /> PAINTING: DRAG ON INDIA · SHIFT-DRAG ERASES
                    <button onClick={() => m.setPaintMode(false)} className="border border-shock bg-shock px-1.5 py-0.5 text-white">DONE</button>
                  </motion.div>
                )}
                {queued > 0 && !leftOpen && (
                  <motion.button
                    key="branch"
                    initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                    whileTap={{ scale: 0.97 }}
                    onClick={() => void m.run()}
                    disabled={m.running}
                    className="pointer-events-auto flex items-center gap-1.5 border border-shock bg-shock px-3 py-1.5 font-pixel text-[9px] text-white shadow-[3px_3px_0_rgba(51,32,15,0.2)] hover:brightness-110 disabled:opacity-60"
                  >
                    <GitBranch size={12} className={m.running ? "mm-pulse" : ""} />
                    {m.running ? "SIMULATING…" : `BRANCH TIMELINE (${queued})`}
                  </motion.button>
                )}
              </AnimatePresence>
            </div>

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

        {/* ── right drawer: impact + cell inspector (IMP-4b) ── */}
        <Drawer side="right" open={rightOpen} onClose={() => setRightOpen(false)} label="Close impact panel" width={330}>
          <ImpactRail m={m} />
        </Drawer>
      </div>

      <HelpModal open={helpOpen} onClose={() => setHelpOpen(false)} />
    </div>
  );
}

function Drawer({
  side, open, onClose, label, width, children,
}: {
  side: "left" | "right"; open: boolean; onClose: () => void; label: string; width: number; children: React.ReactNode;
}) {
  const off = side === "left" ? -width - 12 : width + 12;
  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            className="absolute inset-0 z-40 bg-void/40"
            onClick={onClose}
          />
          <motion.aside
            initial={{ x: off }} animate={{ x: 0 }} exit={{ x: off }}
            transition={{ type: "spring", stiffness: 420, damping: 38 }}
            className={`absolute inset-y-0 z-50 max-w-[92vw] bg-abyss/95 backdrop-blur-md ${
              side === "left" ? "left-0 border-r" : "right-0 border-l"
            } border-grid-line`}
            style={{ width }}
          >
            <button
              onClick={onClose}
              aria-label={label}
              className="absolute right-2 top-2 z-10 grid size-7 place-items-center border border-grid-line bg-abyss text-ink-dim hover:text-ink"
            >
              <X size={13} />
            </button>
            <div className="h-full pt-9">{children}</div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
}
