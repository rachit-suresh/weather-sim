import { useState } from "react";
import { Activity, Radio, Satellite, Zap, Settings2, Check, X } from "lucide-react";
import GlitchText from "@/components/kokonutui/glitch-text";
import type { Multiverse } from "@/state/useMultiverse";

export default function HeaderBar({ m }: { m: Multiverse }) {
  const [urlOpen, setUrlOpen] = useState(false);
  const [draftUrl, setDraftUrl] = useState(m.backendUrl);

  const applyUrl = (next: string) => {
    setDraftUrl(next);
    m.setBackendUrl(next);
    setUrlOpen(false);
  };

  return (
    <header className="relative z-30 flex h-14 shrink-0 items-center gap-3 border-b border-grid-line bg-abyss/90 px-3 backdrop-blur-md sm:gap-4 sm:px-4">
      <div className="flex items-center gap-2.5">
        <div className="grid size-8 place-items-center border border-prime/40 bg-prime/10 glow-prime">
          <Satellite size={15} className="text-prime" />
        </div>
        <div className="leading-none">
          <GlitchText
            text="MULTIVERSE MAP"
            trigger="idle"
            intensity="subtle"
            accent="#c53d2e"
            className="font-pixel text-[13px] text-ink sm:text-[15px]"
          />
          <div className="mt-1 hidden font-mono text-[9px] text-ink-faint sm:block">
            INDIA 32×36 · 1° CELLS · {m.steps}-DAY BRANCH ROLLOUT
          </div>
        </div>
      </div>

      <div className="mx-1 hidden h-7 w-px bg-grid-line md:block" />

      <span
        className="flex items-center gap-1 border border-branch/60 bg-branch/15 px-1.5 py-0.5 font-pixel text-[8px] text-branch mm-pulse"
        title={`Model: ${m.modelName} — surrogate rollout, not an operational forecast.`}
      >
        <Zap size={9} /> SURROGATE
      </span>

      {m.windows.length > 0 && (
        <div className="hidden items-center gap-1.5 lg:flex">
          <select
            value={m.windowIdx}
            onChange={(e) => m.setWindowIdx(Number(e.target.value))}
            aria-label="Historical input window"
            className="max-w-[175px] border border-grid-line bg-panel px-1.5 py-1 font-mono text-[9px] text-ink outline-none focus:border-prime/60"
          >
            {m.windows.map((w, idx) => (
              <option key={w.name} value={idx}>{w.label}</option>
            ))}
          </select>
          <select
            value={m.steps}
            onChange={(e) => m.setSteps(Number(e.target.value))}
            aria-label="Simulation steps"
            className="border border-grid-line bg-panel px-1.5 py-1 font-mono text-[9px] text-ink outline-none focus:border-prime/60"
          >
            {Array.from({ length: m.maxSteps }, (_, k) => k + 1).map((d) => (
              <option key={d} value={d}>{d}d</option>
            ))}
          </select>
        </div>
      )}

      <div className="ml-auto flex items-center gap-3 sm:gap-4">
        {m.timeline && (
          <div className="hidden text-right font-mono text-[9px] leading-tight text-ink-faint lg:block">
            <div>RUN <span className="text-ink-dim">{m.timeline.runId}</span> · <span className="text-ink-dim">{m.modelName}</span></div>
            <div>
              LATENCY <span className="text-prime">{m.timeline.latencyMs.toFixed(0)} ms</span>
              <span className="ml-1">(srv {m.timeline.inferenceMs.toFixed(0)})</span>
            </div>
          </div>
        )}

        <div className="relative">
          <button
            onClick={() => { setDraftUrl(m.backendUrl); setUrlOpen((o) => !o); }}
            className="hidden items-center gap-1.5 font-mono text-[9px] text-ink-faint hover:text-ink sm:flex"
            title={`Backend: ${m.backendUrl || "same-origin proxy"} (${m.modelName})`}
          >
            <Activity size={11} className={m.online ? "text-safe" : "text-danger"} />
            {m.online ? "LINKED" : "OFFLINE"}
            <Settings2 size={10} />
          </button>

          {urlOpen && (
            <div className="panel absolute right-0 top-full z-50 mt-2 w-[275px] p-2.5 text-[10px]">
              <div className="flex items-center justify-between">
                <span className="font-pixel text-[8px] text-ink-dim">BACKEND URL</span>
                <button onClick={() => setUrlOpen(false)} className="text-ink-faint hover:text-ink" aria-label="Close server settings">
                  <X size={12} />
                </button>
              </div>
              <form className="mt-1.5 flex gap-1" onSubmit={(e) => { e.preventDefault(); applyUrl(draftUrl); }}>
                <input
                  type="text"
                  value={draftUrl}
                  onChange={(e) => setDraftUrl(e.target.value)}
                  placeholder="http://127.0.0.1:8000 (blank = proxy)"
                  className="min-w-0 flex-1 border border-grid-line bg-panel px-2 py-1 font-mono text-[10px] text-ink outline-none focus:border-prime/60"
                />
                <button
                  type="submit"
                  className="grid size-7 shrink-0 place-items-center border border-prime/50 bg-prime/15 text-prime hover:bg-prime/25"
                  aria-label="Apply backend URL"
                >
                  <Check size={12} />
                </button>
              </form>
              <div className="mt-1.5 flex gap-1">
                <button type="button" onClick={() => applyUrl("")} className="border border-grid-line bg-abyss px-1.5 py-0.5 text-[9px] text-ink-dim hover:text-ink">
                  Proxy
                </button>
                <button type="button" onClick={() => applyUrl("http://127.0.0.1:8000")} className="border border-grid-line bg-abyss px-1.5 py-0.5 text-[9px] text-ink-dim hover:text-ink">
                  127.0.0.1:8000
                </button>
              </div>
            </div>
          )}
        </div>

        <div
          className={`flex items-center gap-1.5 border px-2 py-1 font-pixel text-[8px] ${
            m.diverged
              ? "border-branch/50 bg-branch/10 text-branch"
              : "border-prime/40 bg-prime/10 text-prime"
          }`}
        >
          <Radio size={10} className={m.running ? "mm-pulse" : ""} />
          {m.running ? "SIMULATING" : m.diverged ? "BRANCHED" : "PRIME"}
        </div>
      </div>
    </header>
  );
}
