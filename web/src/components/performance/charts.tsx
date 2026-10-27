/**
 * Small, dependency-free SVG charts for the performance section.
 * Conventions (dataviz guide): one colour per model across every chart, thin marks with 4px rounded data ends,
 * hairline solid gridlines, text in ink colours (never the series colour), legend always present, hover details.
 */
import { Fragment, useState } from "react";

import { useElementWidth } from "../../hooks/useElementWidth";
import { pct } from "../../lib/format";
import type { ModelEntry } from "../../types";

export const SHORT_NAMES: Record<string, string> = {
  good: "Good",
  hole: "Hole",
  stain: "Stain",
  broken_thread: "Thread",
};

export function Legend({ items }: { items: { label: string; color: string; faded?: boolean }[] }) {
  return (
    <ul className="flex flex-wrap gap-x-5 gap-y-1.5 text-xs text-ink-2">
      {items.map((it) => (
        <li key={it.label} className="flex items-center gap-2">
          <span className="h-2.5 w-4 rounded-sm" style={{ background: it.color, opacity: it.faded ? 0.45 : 1 }} aria-hidden="true" />
          {it.label}
        </li>
      ))}
    </ul>
  );
}

function barPath(x: number, y: number, w: number, h: number, r: number): string {
  const rr = Math.min(r, w, h / 2);
  return `M${x},${y}H${x + w - rr}Q${x + w},${y} ${x + w},${y + rr}V${y + h - rr}Q${x + w},${y + h} ${x + w - rr},${y + h}H${x}Z`;
}

const METRICS = [
  ["accuracy", "Accuracy"],
  ["precision_macro", "Precision"],
  ["recall_macro", "Recall"],
  ["f1_macro", "F1 score"],
] as const;

/** Horizontal grouped bars: four test metrics, one bar per model. */
export function MetricBars({ models, colorOf }: { models: ModelEntry[]; colorOf: (key: string) => string }) {
  const [ref, width] = useElementWidth<HTMLDivElement>();
  const [hover, setHover] = useState<{ text: string; x: number; y: number } | null>(null);
  const labelW = width < 420 ? 70 : 84;
  const valueW = 50;
  const plotW = Math.max(60, width - labelW - valueW);
  const barH = 14;
  const gap = 2;
  const groupGap = 22;
  const groupH = models.length * barH + (models.length - 1) * gap;
  const top = 6;
  const axisH = 22;
  const height = top + METRICS.length * (groupH + groupGap) - groupGap + 8 + axisH;
  const plotBottom = height - axisH;

  return (
    <div ref={ref} className="relative" onMouseLeave={() => setHover(null)}>
      {width > 0 && (
        <svg width={width} height={height} role="img" aria-label="Test accuracy, precision, recall and F1 score of each model">
          {[0, 0.25, 0.5, 0.75, 1].map((t) => (
            <g key={t}>
              <line x1={labelW + t * plotW} x2={labelW + t * plotW} y1={top - 4} y2={plotBottom} stroke="var(--chart-grid)" strokeWidth={1} />
              <text x={labelW + t * plotW} y={height - 6} textAnchor="middle" fontSize={11} fill="var(--muted)">
                {t * 100}%
              </text>
            </g>
          ))}
          {METRICS.map(([key, label], gi) => {
            const gy = top + gi * (groupH + groupGap);
            return (
              <g key={key}>
                <text x={0} y={gy + groupH / 2} dominantBaseline="middle" fontSize={12} fill="var(--ink-2)">
                  {label}
                </text>
                {models.map((m, mi) => {
                  const v = m.test[key];
                  const y = gy + mi * (barH + gap);
                  const w = Math.max(1, v * plotW);
                  return (
                    <g key={m.key} onMouseEnter={() => setHover({ text: `${m.name}: ${label} ${pct(v)}`, x: labelW + w, y })}>
                      <path d={barPath(labelW, y, w, barH, 4)} fill={colorOf(m.key)} />
                      <rect x={labelW} y={y - 1} width={plotW + valueW} height={barH + 2} fill="transparent" />
                      <text x={labelW + w + 6} y={y + barH / 2} dominantBaseline="middle" fontSize={11} fill="var(--ink)" style={{ fontVariantNumeric: "tabular-nums" }}>
                        {pct(v)}
                      </text>
                    </g>
                  );
                })}
              </g>
            );
          })}
          <line x1={labelW} x2={labelW} y1={top - 4} y2={plotBottom} stroke="var(--chart-axis)" strokeWidth={1} />
        </svg>
      )}
      {hover && <Tooltip x={hover.x} y={hover.y} text={hover.text} />}
    </div>
  );
}

function Tooltip({ x, y, text }: { x: number; y: number; text: string }) {
  return (
    <div
      className="pointer-events-none absolute z-10 -translate-x-1/2 -translate-y-full rounded-lg border border-line bg-surface-2 px-2.5 py-1.5 text-xs whitespace-nowrap text-ink shadow-lg"
      style={{ left: x, top: y - 6 }}
    >
      {text}
    </div>
  );
}

function niceMax(v: number): number {
  const step = Math.pow(10, Math.floor(Math.log10(v)));
  return Math.ceil(v / step) * step;
}

/** Train vs validation curve of one model (small multiple). Same hue as the model; training is drawn faded. */
export function TrainingCurve({ model, color, metric, yMax }: { model: ModelEntry; color: string; metric: "accuracy" | "loss"; yMax: number }) {
  const [ref, width] = useElementWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const train = model.history[metric] ?? [];
  const val = model.history[`val_${metric}`] ?? [];
  const n = Math.max(train.length, val.length);
  const height = 220;
  const m = { l: 42, r: 14, t: 14, b: 30 };
  const iw = Math.max(10, width - m.l - m.r);
  const ih = height - m.t - m.b;
  const top = metric === "accuracy" ? 1 : niceMax(yMax);
  const xs = (i: number) => m.l + (n <= 1 ? iw / 2 : (i / (n - 1)) * iw);
  const ys = (v: number) => m.t + ih - (Math.min(v, top) / top) * ih;
  const path = (vals: number[]) => vals.map((v, i) => `${i ? "L" : "M"}${xs(i)},${ys(v)}`).join("");
  const yTicks = [0, 0.25, 0.5, 0.75, 1].map((t) => t * top);
  const step = Math.max(1, Math.ceil(n / 6));
  const xTicks = Array.from({ length: n }, (_, i) => i).filter((i) => i % step === 0 || i === n - 1);
  const fmt = (v: number) => (metric === "accuracy" ? pct(v, 0) : v.toFixed(2));
  const ft = model.fine_tune_start_epoch;

  if (n === 0) return <p className="text-sm text-muted">No training history recorded.</p>;
  return (
    <div ref={ref} className="relative">
      {width > 0 && (
        <svg
          width={width}
          height={height}
          role="img"
          aria-label={`${model.name}: training and validation ${metric} over ${n} epochs`}
          onMouseMove={(e) => {
            const box = e.currentTarget.getBoundingClientRect();
            const i = Math.round(((e.clientX - box.left - m.l) / iw) * (n - 1));
            setHover(Math.min(n - 1, Math.max(0, i)));
          }}
          onMouseLeave={() => setHover(null)}
        >
          {yTicks.map((t) => (
            <g key={t}>
              <line x1={m.l} x2={m.l + iw} y1={ys(t)} y2={ys(t)} stroke="var(--chart-grid)" strokeWidth={1} />
              <text x={m.l - 8} y={ys(t)} textAnchor="end" dominantBaseline="middle" fontSize={11} fill="var(--muted)">
                {fmt(t)}
              </text>
            </g>
          ))}
          {xTicks.map((i) => (
            <text key={i} x={xs(i)} y={height - 8} textAnchor="middle" fontSize={11} fill="var(--muted)">
              {i + 1}
            </text>
          ))}
          <line x1={m.l} x2={m.l + iw} y1={ys(0)} y2={ys(0)} stroke="var(--chart-axis)" strokeWidth={1} />
          {ft !== null && ft > 0 && ft < n && (
            <g>
              <line x1={xs(ft)} x2={xs(ft)} y1={m.t} y2={m.t + ih} stroke="var(--muted)" strokeWidth={1} />
              <text x={xs(ft) + 4} y={m.t + 10} fontSize={10.5} fill="var(--ink-2)">
                fine-tuning →
              </text>
            </g>
          )}
          <path d={path(train)} fill="none" stroke={color} strokeOpacity={0.45} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
          <path d={path(val)} fill="none" stroke={color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
          {val.length > 0 && <circle cx={xs(val.length - 1)} cy={ys(val[val.length - 1])} r={4} fill={color} stroke="var(--surface-2)" strokeWidth={2} />}
          {hover !== null && (
            <g>
              <line x1={xs(hover)} x2={xs(hover)} y1={m.t} y2={m.t + ih} stroke="var(--ink-2)" strokeOpacity={0.5} strokeWidth={1} />
              {train[hover] !== undefined && <circle cx={xs(hover)} cy={ys(train[hover])} r={4} fill={color} fillOpacity={0.5} stroke="var(--surface-2)" strokeWidth={2} />}
              {val[hover] !== undefined && <circle cx={xs(hover)} cy={ys(val[hover])} r={4} fill={color} stroke="var(--surface-2)" strokeWidth={2} />}
            </g>
          )}
        </svg>
      )}
      {hover !== null && (
        <Tooltip
          x={Math.min(Math.max(xs(hover), 80), width - 80)}
          y={m.t + 8}
          text={`Epoch ${hover + 1} · train ${train[hover] !== undefined ? fmt(train[hover]) : "–"} · validation ${val[hover] !== undefined ? fmt(val[hover]) : "–"}`}
        />
      )}
      <p className="text-center text-[11px] text-muted">Epoch</p>
    </div>
  );
}

/** Confusion matrix as an accessible grid. Colour intensity = share of the actual class (row-normalised). */
export function ConfusionMatrix({
  matrix,
  classNames,
  displayNames,
  normalize,
}: {
  matrix: number[][];
  classNames: string[];
  displayNames: Record<string, string>;
  normalize: boolean;
}) {
  const [active, setActive] = useState<[number, number] | null>(null);
  const rowSums = matrix.map((r) => r.reduce((a, b) => a + b, 0));
  const k = classNames.length;
  const name = (i: number) => displayNames[classNames[i]] ?? classNames[i];
  const detail = active
    ? (() => {
        const [i, j] = active;
        const v = matrix[i][j];
        const share = rowSums[i] ? v / rowSums[i] : 0;
        return i === j
          ? `${name(i)}: ${v} of ${rowSums[i]} test images classified correctly (${pct(share)}).`
          : `Actual ${name(i)} predicted as ${name(j)}: ${v} image${v === 1 ? "" : "s"} (${pct(share)} of that class).`;
      })()
    : "Hover over or focus a cell for details. The diagonal holds the correct predictions.";

  return (
    <div>
      <div className="overflow-x-auto">
        <div className="grid min-w-[380px] gap-[2px]" style={{ gridTemplateColumns: `minmax(52px,auto) repeat(${k}, minmax(40px,1fr))` }}>
          <div className="self-end pr-2 pb-1 text-right text-[10px] text-muted">actual ↓ · predicted →</div>
          {classNames.map((c) => (
            <div key={c} className="pb-1 text-center text-[11px] font-medium text-ink-2" title={displayNames[c]}>
              {SHORT_NAMES[c] ?? c}
            </div>
          ))}
          {matrix.map((row, i) => (
            <Fragment key={classNames[i]}>
              <div className="self-center pr-2 text-right text-[11px] font-medium text-ink-2" title={name(i)}>
                {SHORT_NAMES[classNames[i]] ?? classNames[i]}
              </div>
              {row.map((v, j) => {
                const share = rowSums[i] ? v / rowSums[i] : 0;
                return (
                  <div
                    key={j}
                    tabIndex={0}
                    aria-label={`Actual ${name(i)}, predicted ${name(j)}: ${v}`}
                    onMouseEnter={() => setActive([i, j])}
                    onFocus={() => setActive([i, j])}
                    onMouseLeave={() => setActive(null)}
                    onBlur={() => setActive(null)}
                    className="flex aspect-[1.2] items-center justify-center rounded-[4px] text-xs font-semibold"
                    style={{
                      background: `color-mix(in oklab, var(--chart-1) ${Math.round(5 + 90 * share)}%, var(--surface-2))`,
                      color: share > 0.5 ? "#ffffff" : "var(--ink)",
                      fontVariantNumeric: "tabular-nums",
                      boxShadow: i === j ? "inset 0 0 0 1.5px color-mix(in oklab, var(--ink) 30%, transparent)" : undefined,
                    }}
                  >
                    {normalize ? `${Math.round(share * 100)}%` : v}
                  </div>
                );
              })}
            </Fragment>
          ))}
        </div>
      </div>
      <p className="mt-3 min-h-10 text-sm text-ink-2" aria-live="polite">
        {detail}
      </p>
    </div>
  );
}
