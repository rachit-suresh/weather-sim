import { useRef, useState } from "react";
import { Satellite, Cpu, Settings2, Check, X, Presentation, Upload, Download } from "lucide-react";
import GlitchText from "@/components/kokonutui/glitch-text";
import type { Multiverse } from "@/state/useMultiverse";
import { InputFileError, parseInputFile, windowToJson } from "@/lib/windows";

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
/** spec IMP-4f: "monsoon_onset (input ends 20200513)" -> "Monsoon onset, 13 May 2020" */
function prettyWindow(name: string, label: string): string {
  const words = name.replace(/_/g, " ").trim();
  const nice = words.charAt(0).toUpperCase() + words.slice(1);
  const d = /(\d{4})(\d{2})(\d{2})/.exec(label);
  return d ? `${nice}, ${Number(d[3])} ${MONTHS[Number(d[2]) - 1]} ${d[1]}` : nice;
}

export default function HeaderBar({ m, onHelp }: { m: Multiverse; onHelp: () => void }) {
  const [urlOpen, setUrlOpen] = useState(false);
  const [draftUrl, setDraftUrl] = useState(m.backendUrl);

  const applyUrl = (next: string) => {
    setDraftUrl(next);
    m.setBackendUrl(next);
    setUrlOpen(false);
  };

  /* spec IMP-5: load the user's own 3-day input from a .json file, or save the current one as a template */
  const fileRef = useRef<HTMLInputElement>(null);
  const onFile = async (file: File | undefined) => {
    if (!file) return;
    try {
      m.addCustomWindows(parseInputFile(await file.text(), file.name));
    } catch (e) {
      m.reportInputProblem(e instanceof InputFileError ? e.message : `could not read ${file.name}`);
    } finally {
      if (fileRef.current) fileRef.current.value = "";   // allow loading the same file again
    }
  };
  const saveInput = () => {
    const w = m.activeWindow;
    if (!w) return;
    const blob = new Blob([windowToJson(w)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${w.name.replace(/^Your input: /, "").replace(/[^\w.-]+/g, "_") || "input"}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
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
          <div className="mt-1 hidden font-mono text-[9px] text-ink-faint lg:block">
            WHAT-IF WEATHER FOR INDIA · {m.steps}-DAY FORECAST
          </div>
        </div>
      </div>

      <div className="mx-1 hidden h-7 w-px bg-grid-line md:block" />

      <span
        className={`flex items-center gap-1.5 border px-1.5 py-0.5 font-mono text-[9px] ${
          m.online ? "border-safe/60 bg-safe/10 text-ink" : "border-danger/60 bg-danger/10 text-danger"
        }`}
        title={m.online
          ? `Live model reported by /health: ${m.modelName} (${m.backendUrl || "same-origin proxy"})` +
            (m.timeline ? `
Run ${m.timeline.runId} · latency ${m.timeline.latencyMs.toFixed(0)} ms (server ${m.timeline.inferenceMs.toFixed(0)} ms)` : "")
          : `Backend offline (${m.backendUrl || "same-origin proxy"})`}
        aria-label={m.online ? `Backend online, model ${m.modelName}` : "Backend offline"}
      >
        <span className={`inline-block size-2 rounded-full ${m.online ? "bg-safe mm-pulse" : "bg-danger"}`} />
        <Cpu size={10} />
        {m.online ? m.modelName : "OFFLINE"}
      </span>

      {m.windows.length > 0 && (
        <div className="hidden items-center gap-1.5 sm:flex">
          <select
            value={m.windowIdx}
            onChange={(e) => m.setWindowIdx(Number(e.target.value))}
            aria-label="Historical input window"
            title="Start date of the forecast (real weather up to this day)"
            className="max-w-[150px] md:max-w-[210px] border border-grid-line bg-panel px-1.5 py-1 font-mono text-[9px] text-ink outline-none focus:border-prime/60"
          >
            {m.windows.map((w, idx) => (
              <option key={w.name} value={idx}>{prettyWindow(w.name, w.label)}</option>
            ))}
          </select>
          <select
            value={m.steps}
            onChange={(e) => m.setSteps(Number(e.target.value))}
            aria-label="Simulation steps"
            title="Forecast length in days"
            className="border border-grid-line bg-panel px-1.5 py-1 font-mono text-[9px] text-ink outline-none focus:border-prime/60"
          >
            {Array.from({ length: m.maxSteps }, (_, k) => k + 1).map((d) => (
              <option key={d} value={d}>{d} days</option>
            ))}
          </select>
        </div>
      )}

      <div className="flex items-center gap-1">
        <input
          ref={fileRef}
          type="file"
          accept=".json,application/json"
          className="hidden"
          aria-label="Input file (.json)"
          onChange={(e) => void onFile(e.target.files?.[0])}
        />
        <button
          type="button"
          onClick={() => fileRef.current?.click()}
          className="flex h-7 items-center gap-1.5 border border-branch/60 bg-branch/10 px-2 font-pixel text-[8px] text-branch hover:bg-branch/20"
          title='Start from your own weather: a .json with init_states [3][6][32][36] in physical units (oldest day first) and the last input day as "date": "YYYY-MM-DD"'
          aria-label="Load input"
        >
          <Upload size={12} /> <span className="hidden md:inline">LOAD INPUT</span>
        </button>
        <button
          type="button"
          onClick={saveInput}
          disabled={!m.activeWindow}
          className="grid size-7 place-items-center border border-grid-line bg-abyss text-ink-faint hover:text-ink disabled:opacity-40"
          title="Save the current input as a .json template you can edit and load back"
          aria-label="Save input"
        >
          <Download size={12} />
        </button>
      </div>

      <div className="ml-auto flex items-center gap-3 sm:gap-4">
        <div className="relative">
          <button
            onClick={() => { setDraftUrl(m.backendUrl); setUrlOpen((o) => !o); }}
            className="grid size-7 place-items-center border border-grid-line bg-abyss text-ink-faint hover:text-ink"
            title={`Server settings (backend: ${m.backendUrl || "same-origin proxy"})`}
            aria-label={`Backend: ${m.backendUrl || "same-origin proxy"} (${m.modelName})`}
          >
            <Settings2 size={12} />
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

        <button
          onClick={onHelp}
          className="flex h-7 items-center gap-1.5 border border-prime/50 bg-prime/10 px-2 font-pixel text-[8px] text-prime hover:bg-prime/20"
          aria-label="Presentation slides"
        >
          <Presentation size={12} /> <span className="hidden sm:inline">SLIDES</span>
        </button>
      </div>
    </header>
  );
}
