import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  checkHealth, fetchMeta, runBaseline, runBranch, clearPrimeCache,
  getSavedBackendUrl, saveBackendUrl, SimulateError, specFootprint,
  type Timeline, type PerturbationSpec,
} from "@/lib/api";
import { CELLS, GRID, NC, NR, VAR_KEYS, latLonToCell, rowOf, colOf, idx, displayUnit, type VarKey } from "@/lib/grid";
import { loadDemoWindows, type DemoWindow } from "@/lib/windows";
import type { LensId } from "@/lib/impact";
import type { ViewMode, Ripple } from "@/components/PixelGrid";
import type { IslandTask } from "@/components/kokonutui/status-island";

export interface QueuedShock extends PerturbationSpec {
  id: number;
  label: string;
  /** JudgePreset id when this shock is one part of a multi-field judge preset. */
  group?: string;
}

/** Map overlay for one armed shock: the exact cells it perturbs and a short label like "T2M +3 K". */
export interface ShockOutline {
  key: string;
  cells: Set<number>;
  label: string;
}

export function shockPartLabel(field: VarKey, value: number, valueMode: "abs" | "pct"): string {
  const unit = valueMode === "pct" ? "%" : field === "t2m" ? " K" : ` ${displayUnit(field)}`;
  return `${field.toUpperCase()} ${value > 0 ? "+" : value < 0 ? "−" : "±"}${Math.abs(value)}${unit}`;
}

export interface PaintedShock {
  field: VarKey;
  value: number;
  valueMode: "abs" | "pct";
  mask: Map<number, number>;
}

export interface Preset {
  label: string;
  field: VarKey;
  value: number;
  valueMode: "abs" | "pct";
  center: [number, number];
  radius_cells: number;
  falloff: "gaussian" | "linear" | "hard";
  blurb: string;
}

export const PRESETS: Preset[] = [
  {
    label: "+2 K Bay of Bengal", field: "t2m", value: 2, valueMode: "abs",
    center: latLonToCell(15.5, 88.5),
    radius_cells: 3, falloff: "gaussian", blurb: "Warm SST anomaly in the Bay of Bengal cyclogenesis zone",
  },
  {
    label: "−3 m/s Arabian Sea jet", field: "u10", value: -3, valueMode: "abs",
    center: latLonToCell(15.5, 69.5),
    radius_cells: 3, falloff: "gaussian", blurb: "Weaken the low-level Somali/Arabian Sea jet",
  },
  {
    label: "+12 kg/m² core moisture", field: "tcwv", value: 12, valueMode: "abs",
    center: latLonToCell(21.5, 79.5),
    radius_cells: 3.5, falloff: "gaussian", blurb: "Load column moisture over central India",
  },
  {
    label: "−4 hPa trough deepening", field: "msl", value: -4, valueMode: "abs",
    center: latLonToCell(26.5, 80.5),
    radius_cells: 4, falloff: "gaussian", blurb: "Deepen the monsoon trough over the Gangetic plain",
  },
  {
    label: "+45 mm/day NE burst", field: "tp", value: 45, valueMode: "abs",
    center: latLonToCell(25.5, 91.5),
    radius_cells: 2, falloff: "gaussian", blurb: "Cloudburst-scale rain pulse over the Khasi hills",
  },
];

/* ── official evaluation shocks (same definitions as the judge's scoring kit) ──────────────────
   A box covers every cell whose CENTRE lies inside [lat0, lat1] x [lon0, lon1].
   Percent shocks are taken from the last input day (init_states[2]), exactly like the judge. */
export interface JudgePreset {
  id: string;
  label: string;
  blurb: string;
  box: [number, number, number, number];   // lat0, lat1, lon0, lon1 (degrees)
  shocks: { field: VarKey; value: number; valueMode: "abs" | "pct" }[];
}

export const JUDGE_PRESETS: JudgePreset[] = [
  {
    id: "judge_warm_ocean",
    label: "Warm Bay of Bengal (Judge)",
    blurb: "+3 K on t2m and +15% on tcwv over 8–20°N, 82–94°E (judge's warm_ocean shock)",
    box: [8, 20, 82, 94],
    shocks: [
      { field: "t2m", value: 3, valueMode: "abs" },
      { field: "tcwv", value: 15, valueMode: "pct" },
    ],
  },
  {
    id: "judge_monsoon_stall",
    label: "Monsoon Stall (Judge)",
    blurb: "−40% on u10 and v10 over 8–20°N, 65–75°E (judge's monsoon_stall shock)",
    box: [8, 20, 65, 75],
    shocks: [
      { field: "u10", value: -40, valueMode: "pct" },
      { field: "v10", value: -40, valueMode: "pct" },
    ],
  },
  {
    id: "zero_shock",
    label: "Zero Shock (Sanity Check)",
    blurb: "Sends an all-zero perturbation tensor: the branch must equal Prime exactly (judge's null test)",
    box: [GRID.lat[0], GRID.lat[NR - 1], GRID.lon[0], GRID.lon[NC - 1]],
    shocks: [{ field: "t2m", value: 0, valueMode: "abs" }],
  },
];

export function boxMask([lat0, lat1, lon0, lon1]: JudgePreset["box"]): [number, number, number][] {
  const out: [number, number, number][] = [];
  for (let r = 0; r < NR; r++) {
    for (let c = 0; c < NC; c++) {
      const lat = GRID.lat[r], lon = GRID.lon[c];
      if (lat >= lat0 && lat <= lat1 && lon >= lon0 && lon <= lon1) out.push([r, c, 1]);
    }
  }
  return out;
}

export const MAX_PERTURBATIONS = 24;

/** Prefix of every connection/input error: the UI never substitutes synthetic data. */
export const OFFLINE_BANNER = "Backend offline or demo_windows.json missing";

let shockId = 1;
let rippleId = 1;
let taskId = 1;

export function useMultiverse() {
  const [backendUrl, setBackendUrlState] = useState<string>(() => getSavedBackendUrl());
  const [timeline, setTimeline] = useState<Timeline | null>(null);
  const [running, setRunning] = useState(false);
  const [booted, setBooted] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tasks, setTasks] = useState<IslandTask[]>([]);

  const [windows, setWindows] = useState<DemoWindow[]>([]);
  const [windowIdx, setWindowIdxState] = useState(0);
  const [steps, setStepsState] = useState(10);
  const [maxSteps, setMaxSteps] = useState(14);
  const [landSeaMask, setLandSeaMask] = useState<number[][]>(() => GRID.land_sea_mask);

  const [view, setView] = useState<ViewMode>("branched");
  const [variable, setVariable] = useState<VarKey>("tp");
  const [day, setDay] = useState(0);
  const [lens, setLens] = useState<LensId | null>(null);
  const [playing, setPlaying] = useState(false);

  const [paintMode, setPaintMode] = useState(false);
  const [paintField, setPaintField] = useState<VarKey>("t2m");
  const [paintValue, setPaintValue] = useState(2);
  const [paintValueMode, setPaintValueMode] = useState<"abs" | "pct">("abs");
  const [brushRadius, setBrushRadius] = useState(2);
  const [painted, setPainted] = useState<PaintedShock | null>(null);
  const [queue, setQueue] = useState<QueuedShock[]>([]);

  const [selected, setSelected] = useState<number | null>(null);
  const [ripples, setRipples] = useState<Ripple[]>([]);
  const [showWind, setShowWind] = useState(true);
  const [showCities, setShowCities] = useState(true);
  const [showScenery, setShowScenery] = useState(false);   // data first; scenery is an opt-in toggle
  const [focusRegion, setFocusRegion] = useState<string | null>(null);

  const [modelName, setModelName] = useState("—");
  const [online, setOnline] = useState(false);

  const abortRef = useRef<AbortController | null>(null);
  const reqSeqRef = useRef(0);
  const timelineRef = useRef<Timeline | null>(null);
  timelineRef.current = timeline;

  const activeWindow = windows[windowIdx] ?? null;

  const pushTask = useCallback((t: Omit<IslandTask, "id">) => {
    const id = `t${taskId++}`;
    setTasks((prev) => [...prev.slice(-3), { ...t, id }]);
    return id;
  }, []);
  const patchTask = useCallback((id: string, patch: Partial<IslandTask>) => {
    setTasks((prev) => prev.map((t) => (t.id === id ? { ...t, ...patch } : t)));
  }, []);
  const dismissTask = useCallback((id: string) => {
    setTasks((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const addRipple = useCallback((r: number, c: number, kind: Ripple["kind"]) => {
    setRipples((prev) => [...prev.slice(-4), { id: rippleId++, r, c, kind }]);
  }, []);

  const loadConnectionAndBaseline = useCallback(async (
    url: string,
    win: DemoWindow,
    nSteps: number,
  ) => {
    abortRef.current?.abort();
    const ac = new AbortController();
    abortRef.current = ac;
    const seq = ++reqSeqRef.current;

    setRunning(true);
    setError(null);
    setPlaying(false);
    const tid = pushTask({
      label: "Connecting & loading baseline",
      detail: `${nSteps}-day prime rollout`,
      status: "running",
    });

    try {
      const [h, meta] = await Promise.all([
        checkHealth(url, ac.signal),
        fetchMeta(url, ac.signal),
      ]);
      if (seq !== reqSeqRef.current) return;

      const isReady = h.status === "ready";
      setOnline(isReady);
      setModelName(h.model_name || meta.model_name);
      setMaxSteps(meta.max_steps);
      setLandSeaMask(meta.land_sea_mask);

      if (!isReady) {
        throw new SimulateError(`Backend status is "${h.status}" (expected "ready")`);
      }

      const clampedSteps = Math.min(meta.max_steps, Math.max(1, nSteps));
      if (clampedSteps !== nSteps) setStepsState(clampedSteps);

      const t = await runBaseline({
        baseUrl: url,
        window: win,
        steps: clampedSteps,
        modelName: meta.model_name,
      }, ac.signal);
      if (seq !== reqSeqRef.current) return;

      setTimeline(t);
      setDay((d) => Math.min(d, t.days - 1));
      setBooted(true);
      patchTask(tid, {
        label: "Prime timeline ready",
        detail: `${t.modelName} · ${t.days}d · ${t.inferenceMs.toFixed(0)} ms`,
        status: "done",
      });
    } catch (e) {
      if ((e as Error).name === "AbortError" || seq !== reqSeqRef.current) {
        dismissTask(tid);
        return;
      }
      setOnline(false);
      setBooted(true);
      setModelName("—");
      setTimeline(null);                 // never leave an older run on screen as if it were current
      const msg = e instanceof SimulateError ? e.message : String(e);
      setError(`${OFFLINE_BANNER}: ${msg}`);
      patchTask(tid, {
        label: "Backend error",
        detail: msg,
        status: "error",
        action: { label: "Retry", onClick: () => void loadConnectionAndBaseline(url, win, nSteps) },
      });
    } finally {
      if (seq === reqSeqRef.current) setRunning(false);
    }
  }, [pushTask, patchTask, dismissTask]);

  /* ── boot: load demo windows then connect & run baseline ────────────────── */
  useEffect(() => {
    const ac = new AbortController();
    loadDemoWindows(ac.signal)
      .then((loaded) => {
        setWindows(loaded);
        void loadConnectionAndBaseline(backendUrl, loaded[0], steps);
      })
      .catch((e) => {
        if ((e as Error).name !== "AbortError") {
          setBooted(true);
          setError(`${OFFLINE_BANNER}: ${(e as Error).message || String(e)}. No input windows loaded, nothing is simulated.`);
        }
      });
    return () => ac.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const setBackendUrl = useCallback((nextUrl: string) => {
    const saved = saveBackendUrl(nextUrl);
    setBackendUrlState(saved);
    clearPrimeCache();
    setQueue([]);
    setPainted(null);
    setView("branched");
    if (activeWindow) {
      void loadConnectionAndBaseline(saved, activeWindow, steps);
    }
  }, [activeWindow, steps, loadConnectionAndBaseline]);

  const setWindowIdx = useCallback((nextIdx: number) => {
    const win = windows[nextIdx];
    if (!win) return;
    setWindowIdxState(nextIdx);
    setDay(0);
    setView("branched");
    setQueue([]);
    setPainted(null);
    void loadConnectionAndBaseline(backendUrl, win, steps);
  }, [windows, backendUrl, steps, loadConnectionAndBaseline]);

  const setSteps = useCallback((nextSteps: number) => {
    const clamped = Math.min(Math.min(14, maxSteps), Math.max(1, Math.round(nextSteps)));
    setStepsState(clamped);
    setDay((d) => Math.min(d, clamped - 1));
    if (activeWindow) {
      void loadConnectionAndBaseline(backendUrl, activeWindow, clamped);
    }
  }, [maxSteps, activeWindow, backendUrl, loadConnectionAndBaseline]);

  /* ── day playback ──────────────────────────────────────────────────────── */
  useEffect(() => {
    if (!playing || !timeline) return;
    const iv = setInterval(() => {
      setDay((d) => {
        if (d >= timeline.days - 1) { setPlaying(false); return d; }
        return d + 1;
      });
    }, 850);
    return () => clearInterval(iv);
  }, [playing, timeline]);

  /* ── painting ──────────────────────────────────────────────────────────── */
  const onPaint = useCallback((cells: { i: number; w: number }[], erase: boolean) => {
    setPainted((prev) => {
      const mask = new Map(prev?.mask ?? []);
      for (const { i, w } of cells) {
        if (erase) mask.delete(i);
        else mask.set(i, Math.max(mask.get(i) ?? 0, w));
      }
      if (mask.size === 0) return null;
      return { field: paintField, value: paintValue, valueMode: paintValueMode, mask };
    });
  }, [paintField, paintValue, paintValueMode]);

  const onPaintEnd = useCallback(() => { /* accumulated locally; submitted on branch */ }, []);

  const setField = useCallback((f: VarKey) => {
    setPaintField(f);
    setPainted((prev) => (prev ? { ...prev, field: f } : prev));
  }, []);

  const setDeltaValue = useCallback((v: number) => {
    setPaintValue(v);
    setPainted((prev) => (prev ? { ...prev, value: v } : prev));
  }, []);

  const setValueMode = useCallback((mode: "abs" | "pct") => {
    setPaintValueMode(mode);
    setPaintValue((prev) => (mode === "pct" ? Math.max(-50, Math.min(50, Math.round(prev) || 10)) : prev));
    setPainted((prev) => (prev ? { ...prev, valueMode: mode } : prev));
  }, []);

  const clearPaint = useCallback(() => setPainted(null), []);

  const perturbationCount = queue.length + (painted ? 1 : 0);

  const commitPaintedToQueue = useCallback(() => {
    if (!painted || !painted.mask.size) return;
    if (queue.length >= MAX_PERTURBATIONS) return;
    const unit = painted.valueMode === "pct" ? "%" : GRID.units[painted.field] === "Pa" ? "hPa" : GRID.units[painted.field];
    const sign = painted.value > 0 ? "+" : "";
    const label = `${painted.field.toUpperCase()} ${sign}${painted.value}${unit} (${painted.mask.size}c)`;
    const mask: [number, number, number][] = [...painted.mask.entries()].map(
      ([i, w]) => [rowOf(i), colOf(i), +w.toFixed(3)],
    );
    setQueue((prev) => {
      if (prev.length >= MAX_PERTURBATIONS) return prev;
      return [...prev, {
        id: shockId++,
        label,
        field: painted.field,
        value: painted.value,
        valueMode: painted.valueMode,
        mask,
      }];
    });
    setPainted(null);
  }, [painted, queue.length]);

  const addPreset = useCallback((p: Preset) => {
    setQueue((prev) => {
      if (prev.some((q) => q.label === p.label)) return prev;
      if (prev.length + (painted ? 1 : 0) >= MAX_PERTURBATIONS) return prev;
      return [...prev, {
        id: shockId++,
        label: p.label,
        field: p.field,
        value: p.value,
        valueMode: p.valueMode,
        center: p.center,
        radius_cells: p.radius_cells,
        falloff: p.falloff,
      }];
    });
    addRipple(p.center[0], p.center[1], "select");
  }, [addRipple, painted]);

  const removeShock = useCallback((id: number) => {
    setQueue((prev) => {
      const target = prev.find((q) => q.id === id);
      // parts of a judge preset are only meaningful together: removing one removes the whole preset
      return target?.group ? prev.filter((q) => q.group !== target.group) : prev.filter((q) => q.id !== id);
    });
  }, []);

  const addJudgePreset = useCallback((p: JudgePreset) => {
    const mask = boxMask(p.box);
    const r0 = Math.round(mask.reduce((s, [r]) => s + r, 0) / mask.length);
    const c0 = Math.round(mask.reduce((s, [, c]) => s + c, 0) / mask.length);
    setQueue((prev) => {
      if (prev.some((q) => q.group === p.id)) return prev;
      if (prev.length + p.shocks.length + (painted ? 1 : 0) > MAX_PERTURBATIONS) return prev;
      return [...prev, ...p.shocks.map((s) => {
        const unit = s.valueMode === "pct" ? "%" : GRID.units[s.field] === "Pa" ? "hPa" : GRID.units[s.field];
        return {
          id: shockId++,
          group: p.id,
          label: `${p.label} · ${s.field.toUpperCase()} ${s.value > 0 ? "+" : ""}${s.value}${unit}`,
          field: s.field,
          value: s.value,
          valueMode: s.valueMode,
          mask,
          center: [r0, c0] as [number, number],
        };
      })];
    });
    addRipple(r0, c0, "select");
  }, [addRipple, painted]);

  const removeJudgePreset = useCallback((presetId: string) => {
    setQueue((prev) => prev.filter((q) => q.group !== presetId));
  }, []);

  /* ── the live POST /simulate branch call ───────────────────────────────── */
  const run = useCallback(async () => {
    if (!activeWindow || running) return;
    const specs: PerturbationSpec[] = queue.map(({ id, label, group, ...p }) => {
      void id; void label; void group;
      return p;
    });
    if (painted && painted.mask.size) {
      specs.push({
        field: painted.field,
        value: painted.value,
        valueMode: painted.valueMode,
        mask: [...painted.mask.entries()].map(([i, w]) => [rowOf(i), colOf(i), +w.toFixed(3)]),
      });
    }
    if (!specs.length) return;
    // the painted shock is the newest intent; otherwise the first armed shock (t2m for warm Bay)
    const shockedField: VarKey = painted && painted.mask.size ? painted.field : specs[0].field;

    abortRef.current?.abort();
    const ac = new AbortController();
    abortRef.current = ac;
    const seq = ++reqSeqRef.current;

    setRunning(true);
    setError(null);
    setPlaying(false);
    const tid = pushTask({
      label: "Branching timeline",
      detail: `${specs.length} shock${specs.length > 1 ? "s" : ""} · POST /simulate`,
      status: "running",
    });

    let er = Math.floor(NR / 2), ec = Math.floor(NC / 2);
    if (painted && painted.mask.size) {
      let sr = 0, sc = 0;
      for (const i of painted.mask.keys()) { sr += rowOf(i); sc += colOf(i); }
      er = Math.round(sr / painted.mask.size);
      ec = Math.round(sc / painted.mask.size);
    } else if (queue[0]?.center) {
      [er, ec] = queue[0].center;
    }

    try {
      const t = await runBranch({
        baseUrl: backendUrl,
        window: activeWindow,
        steps,
        modelName,
        specs,
      }, ac.signal);
      if (seq !== reqSeqRef.current) return;

      // largest |branched - prime| over every channel, day and cell (raw backend units)
      let maxAbsDiff = 0;
      for (const v of VAR_KEYS) {
        const b = t.branched[v], p = t.prime[v];
        for (let i = 0; i < b.length; i++) {
          const d = Math.abs(b[i] - p[i]);
          if (d > maxAbsDiff) maxAbsDiff = d;
        }
      }

      setOnline(true);
      setTimeline(t);
      setDay(0);
      // Show the field that was actually shocked, so the difference map is never blank because the
      // viewer happened to be on a channel the shock does not touch (e.g. rain for a +3 K shock).
      setVariable(shockedField);
      setView("diff");
      setPlaying(true);
      addRipple(er, ec, "shock");
      patchTask(tid, {
        label: maxAbsDiff === 0 ? "Branch == Prime (zero change)" : "Branch resolved",
        detail: `${t.inferenceMs.toFixed(0)} ms infer · ${t.runId} · max |Δ| = ${maxAbsDiff === 0 ? "0" : maxAbsDiff.toPrecision(3)}`,
        status: "done",
        action: { label: "Replay", onClick: () => { setDay(0); setPlaying(true); } },
      });
    } catch (e) {
      if ((e as Error).name === "AbortError" || seq !== reqSeqRef.current) {
        dismissTask(tid);
        return;
      }
      const msg = e instanceof SimulateError ? e.message : String(e);
      const unreachable = e instanceof SimulateError && e.status === undefined && msg.startsWith("Cannot reach");
      if (unreachable) setOnline(false);
      setError(unreachable ? `${OFFLINE_BANNER}: ${msg}` : msg);
      patchTask(tid, {
        label: "Branch failed",
        detail: msg,
        status: "error",
        action: { label: "Retry", onClick: () => void run() },
      });
    } finally {
      if (seq === reqSeqRef.current) setRunning(false);
    }
  }, [activeWindow, running, queue, painted, backendUrl, steps, modelName, pushTask, patchTask, dismissTask, addRipple]);

  /* ── reset to prime ────────────────────────────────────────────────────── */
  const reset = useCallback(() => {
    abortRef.current?.abort();
    reqSeqRef.current++;
    setRunning(false);
    setQueue([]);
    setPainted(null);
    setPlaying(false);
    setDay(0);
    setView("branched");
    setError(null);
    const t = timelineRef.current;
    if (t) {
      setTimeline({ ...t, branched: t.prime, runId: `${t.runId}_prime` });
      addRipple(Math.floor(NR / 2), Math.floor(NC / 2), "run");
      pushTask({ label: "Timelines re-converged", detail: "branched ← prime", status: "done" });
    }
  }, [addRipple, pushTask]);

  const selectCell = useCallback((i: number | null) => {
    setSelected(i);
    if (i !== null) addRipple(rowOf(i), colOf(i), "select");
  }, [addRipple]);

  /* ── derived: check divergence across ALL 6 channels ───────────────────── */
  const diverged = useMemo(() => {
    if (!timeline) return false;
    for (const v of VAR_KEYS) {
      const b = timeline.branched[v], p = timeline.prime[v];
      const eps = v === "msl" ? 1e-2 : 1e-5;
      for (let i = 0; i < b.length; i++) {
        if (Math.abs(b[i] - p[i]) > eps) return true;
      }
    }
    return false;
  }, [timeline]);

  /* ── derived: outline + label for every armed shock (judge presets merged into one box) ── */
  const shockOutlines = useMemo<ShockOutline[]>(() => {
    const groups = new Map<string, { cells: Set<number>; parts: string[] }>();
    const add = (key: string, spec: Pick<PerturbationSpec, "field" | "value" | "valueMode" | "mask" | "center" | "radius_cells" | "falloff">) => {
      const g = groups.get(key) ?? { cells: new Set<number>(), parts: [] };
      for (const i of specFootprint(spec).keys()) g.cells.add(i);
      g.parts.push(shockPartLabel(spec.field, spec.value, spec.valueMode));
      groups.set(key, g);
    };
    for (const q of queue) add(q.group ?? `q${q.id}`, q);
    if (painted && painted.mask.size) {
      add("painted", {
        field: painted.field, value: painted.value, valueMode: painted.valueMode,
        mask: [...painted.mask.entries()].map(([i, w]) => [rowOf(i), colOf(i), w] as [number, number, number]),
      });
    }
    return [...groups.entries()]
      .filter(([, g]) => g.cells.size > 0)
      .map(([key, g]) => ({ key, cells: g.cells, label: g.parts.join(" · ") }));
  }, [queue, painted]);

  const missionStage: "paint" | "branch" | "explore" = diverged
    ? "explore"
    : perturbationCount > 0
      ? "branch"
      : "paint";

  return {
    backendUrl, setBackendUrl,
    windows, windowIdx, setWindowIdx, activeWindow,
    steps, setSteps, maxSteps, landSeaMask,
    timeline, running, booted, error, tasks, dismissTask,
    view, setView, variable, setVariable, day, setDay, lens, setLens, playing, setPlaying,
    paintMode, setPaintMode, paintField, setField,
    paintValue, setDeltaValue, paintValueMode, setValueMode,
    brushRadius, setBrushRadius, painted, onPaint, onPaintEnd, clearPaint, commitPaintedToQueue,
    queue, addPreset, removeShock, addJudgePreset, removeJudgePreset, perturbationCount,
    run, reset, selected, selectCell, ripples, addRipple,
    showWind, setShowWind, showCities, setShowCities, showScenery, setShowScenery, focusRegion, setFocusRegion,
    modelName, online, diverged, missionStage, shockOutlines,
  };
}

export type Multiverse = ReturnType<typeof useMultiverse>;
export { CELLS, NC, idx };
