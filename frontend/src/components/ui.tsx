import { useRef } from "react";
import { motion } from "motion/react";
import type { LucideIcon } from "lucide-react";
import SplitFlapText from "@/components/kokonutui/split-flap-text";
import { cn } from "@/lib/utils";

/** Split-flap board in a layout box that matches its scaled size. */
export function Flap({
  text, length, scale = 0.62, className,
}: {
  text: string; length: number; scale?: number; className?: string;
}) {
  const w = Math.ceil(length * 0.8 * 48 * scale);
  const h = Math.ceil(1.22 * 48 * scale);
  return (
    <div className={cn("relative shrink-0 overflow-visible", className)} style={{ width: w, height: h }}>
      <div className="origin-top-left" style={{ transform: `scale(${scale})` }}>
        <SplitFlapText text={text} length={length} />
      </div>
    </div>
  );
}

/** Horizontal pixel scrubber: drag or click across discrete detents. */
export function PixelScrubber({
  value, min, max, onChange, accent = "var(--color-prime)", labels,
}: {
  value: number; min: number; max: number;
  onChange: (v: number) => void;
  accent?: string;
  labels?: (v: number) => string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);
  const n = max - min + 1;

  const setFromX = (clientX: number) => {
    const el = ref.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const frac = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    onChange(Math.round(min + frac * (max - min)));
  };

  return (
    <div
      ref={ref}
      role="slider"
      aria-label="Forecast day"
      aria-valuemin={min}
      aria-valuemax={max}
      aria-valuenow={value}
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "ArrowLeft") onChange(Math.max(min, value - 1));
        if (e.key === "ArrowRight") onChange(Math.min(max, value + 1));
        if (e.key === "Home") onChange(min);
        if (e.key === "End") onChange(max);
      }}
      onPointerDown={(e) => {
        dragging.current = true;
        (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
        setFromX(e.clientX);
      }}
      onPointerMove={(e) => { if (dragging.current) setFromX(e.clientX); }}
      onPointerUp={() => { dragging.current = false; }}
      onPointerCancel={() => { dragging.current = false; }}
      className="relative flex h-9 w-full cursor-ew-resize touch-none select-none border border-grid-line bg-abyss/80 outline-none focus-visible:border-prime/60"
    >
      {Array.from({ length: n }, (_, k) => {
        const v = min + k;
        const active = v === value;
        const passed = v <= value;
        return (
          <div key={v} className="relative flex-1 border-r border-grid-line/60 last:border-r-0">
            <div
              className="absolute inset-y-0 left-0 transition-all duration-200"
              style={{ width: passed ? "100%" : "0%", background: passed ? `${accent}22` : "transparent" }}
            />
            {labels && (
              <span className={`absolute inset-0 grid place-items-center font-pixel text-[7px] ${active ? "text-void" : "text-ink-faint"}`}>
                {labels(v)}
              </span>
            )}
          </div>
        );
      })}
      <motion.div
        className="pointer-events-none absolute inset-y-0"
        animate={{ left: `${((value - min) / n) * 100}%`, width: `${100 / n}%` }}
        transition={{ type: "spring", stiffness: 700, damping: 42 }}
        style={{ background: accent, boxShadow: `0 0 14px -2px ${accent}` }}
      />
      {labels && (
        <span className="pointer-events-none absolute inset-y-0 grid place-items-center font-pixel text-[7px] text-void"
          style={{ left: `${((value - min) / n) * 100}%`, width: `${100 / n}%` }}>
          {labels(value)}
        </span>
      )}
    </div>
  );
}

export interface TabItem {
  id: string;
  label: string;
  icon?: LucideIcon;
  accent?: string;
}

export function PixelTabs({
  items, value, onChange, className, size = "md",
}: {
  items: TabItem[];
  value: string;
  onChange: (id: string) => void;
  className?: string;
  size?: "sm" | "md";
}) {
  return (
    <div className={cn("inline-flex gap-0.5 border border-grid-line bg-abyss/80 p-0.5", className)} role="tablist">
      {items.map((it) => {
        const active = it.id === value;
        const Icon = it.icon;
        return (
          <button
            key={it.id}
            role="tab"
            aria-selected={active}
            onClick={() => onChange(it.id)}
            className={cn(
              "relative font-pixel transition-colors",
              size === "sm" ? "px-2 py-1 text-[8px]" : "px-2.5 py-1.5 text-[9px]",
              active ? "text-void" : "text-ink-dim hover:text-ink",
            )}
          >
            {active && (
              <motion.span
                layoutId={`ptab-${items.map((i) => i.id).join("")}`}
                className="absolute inset-0"
                style={{ background: it.accent ?? "var(--color-prime)" }}
                transition={{ type: "spring", stiffness: 600, damping: 38 }}
              />
            )}
            <span className="relative z-10 flex items-center gap-1">
              {Icon && <Icon size={size === "sm" ? 9 : 11} strokeWidth={2.5} />}
              {it.label}
            </span>
          </button>
        );
      })}
    </div>
  );
}

export function SectionTitle({ children, right }: { children: React.ReactNode; right?: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-2 px-3 pt-3 pb-1.5">
      <span className="font-pixel text-[9px] tracking-wider text-ink-faint">{children}</span>
      {right}
    </div>
  );
}

export function ToggleChip({
  active, onClick, icon: Icon, children, accent = "var(--color-prime)",
}: {
  active: boolean;
  onClick: () => void;
  icon?: LucideIcon;
  children: React.ReactNode;
  accent?: string;
}) {
  return (
    <button
      onClick={onClick}
      aria-pressed={active}
      className={cn(
        "flex items-center gap-1.5 border px-2 py-1 font-pixel text-[8px] transition-all",
        active ? "border-transparent text-void" : "border-grid-line bg-abyss/60 text-ink-dim hover:text-ink",
      )}
      style={active ? { background: accent } : undefined}
    >
      {Icon && <Icon size={10} strokeWidth={2.5} />}
      {children}
    </button>
  );
}

export function StatRow({ label, value, unit, tone }: { label: string; value: string; unit?: string; tone?: string }) {
  return (
    <div className="flex items-baseline justify-between gap-2 py-0.5">
      <span className="text-[10px] text-ink-faint">{label}</span>
      <span className="font-mono text-[11px] tabular-nums" style={{ color: tone ?? "var(--color-ink)" }}>
        {value}{unit && <span className="ml-0.5 text-[9px] text-ink-faint">{unit}</span>}
      </span>
    </div>
  );
}