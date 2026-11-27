import { useCallback, useEffect, useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import { X, ChevronLeft, ChevronRight, Zap, GitBranch, BarChart3 } from "lucide-react";

/** spec IMP-6: full-screen presentation deck (big words, one idea per slide), opened from the header's Slides button.
 *  Every number is a measured value from the submitted model (demo/final_demo_script). */

type Stat = { v: string; l: string };
type Slide = {
  kicker: string;
  title: string;
  sub?: string;
  points?: string[];
  stats?: Stat[];
  flow?: string[];
  tone?: "prime" | "branch" | "shock" | "safe";
  howto?: boolean;
};

const SLIDES: Slide[] = [
  { kicker: "Team Sage · Multiverse of Monsoon", title: "A what-if weather twin for India",
    sub: "Poke the weather. Watch the branch.", tone: "prime" },
  { kicker: "The problem", title: "A forecast shows one future",
    points: ["We let you ask: what if?", "Warmer sea, drier air, hotter week", "Normal and shocked forecasts, side by side"], tone: "branch" },
  { kicker: "Built for the edge", title: "Offline. No GPU. A normal laptop.",
    points: ["Works where connectivity doesn't", "A teaching tool for how the monsoon behaves", "Forecasts become decisions: flood, heat, crops & power"], tone: "safe" },
  { kicker: "The data", title: "India as a 6-layer picture",
    stats: [{ v: "32 × 36", l: "cells, 1° each" }, { v: "6", l: "variables per cell" }, { v: "17 yrs", l: "2005–2021, 6,209 days" }, { v: "3 days", l: "in → up to 14 out" }] },
  { kicker: "Unique idea 1 · representation", title: "Learn the surprise, not the season",
    points: ["Every value = 17-year normal + anomaly", "The network only predicts the anomaly", "Rain on a square-root scale", "29 input maps: 3 days × 6 + target-day normal + land, height, date, edge"], tone: "prime" },
  { kicker: "Architecture", title: "A small residual CNN",
    flow: ["29 maps", "3×3 conv", "3 residual blocks", "Δ tomorrow + rain gate"],
    stats: [{ v: "7", l: "conv layers, 3×3" }, { v: "239k", l: "weights · ≈1 MB" }, { v: "×14", l: "days, one step at a time" }],
    sub: "Last layer starts at zero: untrained, it already says “tomorrow = today”." },
  { kicker: "Unique idea 2 · locality by design", title: "7 layers = 7 cells = one day of weather",
    stats: [{ v: "2.5–2.9", l: "cells hold 95% of a shock (day 1)" }, { v: "9.9", l: "farthest cell that moves" }, { v: "11", l: "judges' limit" }],
    sub: "A shock in Kolkata cannot change Mumbai tomorrow. Whole-map models could, and did.", tone: "branch" },
  { kicker: "Unique idea 3 · rain", title: "Two questions per cell",
    points: ["Does it rain?  →  hard yes / no gate", "How much?  →  only where it rains", "No drizzle smeared everywhere", "Spec bug caught: soft gate in training, hard in use. Fixed with a straight-through gate"], tone: "prime" },
  { kicker: "Training", title: "Learn to survive your own mistakes",
    flow: ["unroll 1 day", "2 days", "4 days", "8 days"],
    points: ["Graded on chains of its own predictions", "3× weight on heavy rain (> 10 mm)", "Spread + spectral losses: no blur by day 8", "Latitude-weighted error, like the judges"], tone: "shock" },
  { kicker: "Model selection", title: "9 families. 30+ models. One survived.",
    points: ["CNN · U-Net · ConvGRU · CNN+Transformer · ViT · GNN · Fourier · Physics hybrid · LightGBM", "Scored with the organisers' own code", "Picked the one that held up on unseen years"],
    stats: [{ v: "0.118", l: "CNN skill, unseen 2020–21" }, { v: "0.104–0.107", l: "the others" }] },
  { kicker: "Unique idea 4 · stability guard", title: "Train with the stress test",
    points: ["Fine-tune while applying the warm-Bay shock", "Penalise rain responses above 3.5σ (limit 5σ)"],
    stats: [{ v: "5.40 → 3.85σ", l: "worst rain response" }, { v: "−0.002", l: "skill cost" }], tone: "safe" },
  { kicker: "Unique idea 5 · cyclone mode", title: "Storms have to move",
    flow: ["deep sea-low?", "track it", "moving circle", "mover: pressure + winds"],
    points: ["Plain CNN kept cyclones parked in place", "Mover carries weather along a learned wind (advect & correct)", "Base CNN keeps rain, temperature, moisture · fires on 2–3% of days"],
    stats: [{ v: "312 → 104 km", l: "Fani centre error, day 2" }], tone: "shock" },
  { kicker: "Engineering", title: "Never fails, never phones home",
    stats: [{ v: "≈30 ms", l: "per 10-day forecast" }, { v: "253 MB", l: "memory" }, { v: "15 / 15", l: "repo scan" }, { v: "0", l: "downloads at run time" }],
    points: ["Fallback model if anything breaks", "Load your own 3-day input as JSON"] },
  { kicker: "How we worked", title: "Spec first. Measure everything.",
    points: ["17 spec versions, written before the code", "Every bug labelled: code error or spec error", "Every claim measured on years the model never saw"], tone: "branch" },
  { kicker: "Results · unseen 2020–21", title: "12–16% lower error than the best free forecast",
    stats: [{ v: "0.116 / 0.163", l: "error skill: ordinary / extremes" }, { v: "0.104 / 0.113", l: "heavy-rain skill" }, { v: "exact", l: "zero shock = no shock" }], tone: "prime" },
  { kicker: "Honest limits", title: "What's next",
    points: ["Monsoon onset: right direction, wrong timing", "Cyclone rain can trail the moving storm", "No uncertainty estimate yet", "Warmer climate than our 17-year normal"], tone: "shock" },
  { kicker: "Team Sage", title: "Ask it a what-if.", sub: "Thank you.", tone: "prime" },
  { kicker: "Using the map", title: "Shock → Branch → Read", howto: true },
];

const TONE: Record<NonNullable<Slide["tone"]>, string> = {
  prime: "text-prime", branch: "text-branch", shock: "text-shock", safe: "text-safe",
};

export default function HelpModal({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [i, setI] = useState(0);
  const n = SLIDES.length;
  const go = useCallback((d: number) => setI((k) => Math.min(n - 1, Math.max(0, k + d))), [n]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      else if (["ArrowRight", "PageDown", " ", "Enter"].includes(e.key)) { e.preventDefault(); go(1); }
      else if (["ArrowLeft", "PageUp", "Backspace"].includes(e.key)) { e.preventDefault(); go(-1); }
      else if (e.key === "Home") setI(0);
      else if (e.key === "End") setI(n - 1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, go, onClose, n]);

  const s = SLIDES[i];
  const accent = s.tone ? TONE[s.tone] : "text-ink";

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
          className="fixed inset-0 z-[80] flex flex-col bg-abyss"
          role="dialog" aria-modal="true" aria-label="Presentation slides"
        >
          <div className="flex items-center justify-between border-b border-grid-line px-4 py-2 font-mono text-[11px] text-ink-faint">
            <span>TEAM SAGE · MULTIVERSE OF MONSOON</span>
            <div className="flex items-center gap-3">
              <span aria-live="polite">{i + 1} / {n}</span>
              <button onClick={onClose} aria-label="Close slides" className="grid size-8 place-items-center border border-grid-line bg-panel text-ink-dim hover:text-ink">
                <X size={15} />
              </button>
            </div>
          </div>

          <div className="relative flex min-h-0 flex-1 items-center justify-center overflow-y-auto px-5 py-6 sm:px-12">
            <AnimatePresence mode="wait">
              <motion.section
                key={i}
                initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.22 }}
                className="w-full max-w-[1100px]"
              >
                <div className={`font-mono text-[clamp(12px,1.4vw,17px)] uppercase tracking-[0.14em] ${accent}`}>{s.kicker}</div>
                <h2 className="mt-3 font-pixel text-[clamp(26px,4.6vw,60px)] leading-[1.15] text-ink">{s.title}</h2>
                {s.sub && <p className="mt-4 text-[clamp(16px,2vw,26px)] leading-snug text-ink-dim">{s.sub}</p>}

                {s.flow && (
                  <div className="mt-7 flex flex-wrap items-center gap-2 sm:gap-3">
                    {s.flow.map((f, k) => (
                      <div key={f} className="flex items-center gap-2 sm:gap-3">
                        <span className="border border-grid-line bg-panel px-3 py-2 font-mono text-[clamp(13px,1.5vw,19px)] text-ink">{f}</span>
                        {k < s.flow!.length - 1 && <ChevronRight size={20} className={accent} />}
                      </div>
                    ))}
                  </div>
                )}

                {s.points && (
                  <ul className="mt-7 space-y-3">
                    {s.points.map((p) => (
                      <li key={p} className="flex gap-3 text-[clamp(17px,2.1vw,28px)] leading-snug text-ink">
                        <span className={`mt-[0.55em] inline-block size-2.5 shrink-0 ${s.tone ? "bg-current " + accent : "bg-ink-dim"}`} />
                        <span>{p}</span>
                      </li>
                    ))}
                  </ul>
                )}

                {s.stats && (
                  <div className="mt-8 grid grid-cols-2 gap-3 md:grid-cols-4">
                    {s.stats.map((st) => (
                      <div key={st.l} className="border border-grid-line bg-panel px-4 py-3">
                        <div className={`font-pixel text-[clamp(20px,2.8vw,38px)] leading-tight ${accent}`}>{st.v}</div>
                        <div className="mt-1 font-mono text-[clamp(11px,1.15vw,14px)] text-ink-dim">{st.l}</div>
                      </div>
                    ))}
                  </div>
                )}

                {s.howto && (
                  <ol className="mt-8 space-y-4 text-[clamp(16px,1.9vw,24px)] leading-snug text-ink">
                    <li className="flex gap-3"><Zap className="mt-1 shrink-0 text-shock" /><span><b>Pick a shock.</b> Open <b className="text-shock">SHOCKS</b>: judge presets, more presets, or paint your own on the map.</span></li>
                    <li className="flex gap-3"><GitBranch className="mt-1 shrink-0 text-branch" /><span><b>Branch the timeline.</b> The shocked forecast runs next to the normal one, then plays the days.</span></li>
                    <li className="flex gap-3"><BarChart3 className="mt-1 shrink-0 text-prime" /><span><b>Read the result.</b> <b className="text-branch">IMPACT</b> for flood, heat, crops & power; click any cell for its readout.</span></li>
                    <li className="font-mono text-[clamp(12px,1.3vw,15px)] text-ink-dim">PRIME = normal · BRANCHED = after your shock · DIFF = only what the shock changed · LOAD INPUT = your own 3 days</li>
                  </ol>
                )}
              </motion.section>
            </AnimatePresence>
          </div>

          <div className="flex items-center justify-between gap-3 border-t border-grid-line px-4 py-3">
            <button onClick={() => go(-1)} disabled={i === 0} aria-label="Previous slide"
              className="flex items-center gap-1 border border-grid-line bg-panel px-3 py-1.5 font-mono text-[11px] text-ink disabled:opacity-30">
              <ChevronLeft size={14} /> PREV
            </button>
            <div className="flex flex-wrap justify-center gap-1.5" role="tablist" aria-label="Slides">
              {SLIDES.map((sl, k) => (
                <button key={sl.title} onClick={() => setI(k)} aria-label={`Slide ${k + 1}: ${sl.title}`} aria-selected={k === i} role="tab"
                  className={`size-2.5 border border-grid-line ${k === i ? "bg-prime" : "bg-panel hover:bg-ink-faint"}`} />
              ))}
            </div>
            <button onClick={() => go(1)} disabled={i === n - 1} aria-label="Next slide"
              className="flex items-center gap-1 border border-grid-line bg-panel px-3 py-1.5 font-mono text-[11px] text-ink disabled:opacity-30">
              NEXT <ChevronRight size={14} />
            </button>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
