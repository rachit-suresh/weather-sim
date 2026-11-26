import { motion, AnimatePresence } from "motion/react";
import { X, Zap, GitBranch, BarChart3 } from "lucide-react";

/** spec IMP-4e: short how-it-works modal, opened from the "?" button in the header. */
export default function HelpModal({ open, onClose }: { open: boolean; onClose: () => void }) {
  return (
    <AnimatePresence>
      {open && (
        <motion.div
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
          className="absolute inset-0 z-[60] grid place-items-center bg-void/70 p-4 backdrop-blur-sm"
          onClick={onClose}
          role="dialog"
          aria-modal="true"
          aria-label="How it works"
        >
          <motion.div
            initial={{ y: 12, scale: 0.98 }} animate={{ y: 0, scale: 1 }} exit={{ y: 8, opacity: 0 }}
            className="panel relative w-[min(460px,100%)] p-4"
            onClick={(e) => e.stopPropagation()}
          >
            <button onClick={onClose} aria-label="Close help" className="absolute right-2 top-2 grid size-7 place-items-center border border-grid-line bg-abyss text-ink-dim hover:text-ink">
              <X size={13} />
            </button>
            <h2 className="font-pixel text-[11px] text-ink">How it works</h2>
            <p className="mt-1.5 text-[11px] leading-relaxed text-ink-dim">
              A neural model forecasts India&apos;s weather 10 days ahead from a real start date. You change the starting
              weather somewhere (a &quot;shock&quot;) and see how the next 10 days turn out differently.
            </p>
            <ol className="mt-3 space-y-2.5 text-[11px] leading-relaxed text-ink-dim">
              <li className="flex gap-2.5">
                <Zap size={14} className="mt-0.5 shrink-0 text-shock" />
                <span><b className="text-ink">Pick a shock.</b> Open <b className="text-shock">SHOCKS</b> and choose one (the judge shocks are at the top), or paint your own on the map.</span>
              </li>
              <li className="flex gap-2.5">
                <GitBranch size={14} className="mt-0.5 shrink-0 text-branch" />
                <span><b className="text-ink">Branch the timeline.</b> Press <b className="text-shock">BRANCH TIMELINE</b>. The model runs the shocked forecast next to the normal one, then plays the 10 days.</span>
              </li>
              <li className="flex gap-2.5">
                <BarChart3 size={14} className="mt-0.5 shrink-0 text-prime" />
                <span><b className="text-ink">Read the result.</b> Open <b className="text-branch">IMPACT</b> for flood, heat and crop/power risk and the places that changed most. Click any cell for its full readout.</span>
              </li>
            </ol>
            <div className="mt-3 border-t border-grid-line pt-2.5 text-[10px] leading-relaxed text-ink-faint">
              <b className="text-prime">PRIME</b> = the normal forecast · <b className="text-branch">BRANCHED</b> = the forecast after your shock ·{" "}
              <b className="text-danger">DIFF</b> = branched minus prime (only what your shock changed).
              Map settings (wind, cities, weather focus) are under <b className="text-ink">VIEW</b>.
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
