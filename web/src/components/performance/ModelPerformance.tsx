import { Award, ChartColumn, Table, TriangleAlert } from "lucide-react";
import { useState } from "react";

import { useSiteData } from "../../hooks/useSiteData";
import { compareModels } from "../../lib/compare";
import { compact, fmtMB, fmtMs, pct } from "../../lib/format";
import type { MetricsFile, ModelEntry } from "../../types";
import { SectionHeader } from "../layout/SectionHeader";
import { ConfusionMatrix, Legend, MetricBars, TrainingCurve } from "./charts";

const ORDER = ["scratch", "transfer"];

function Toggle<T extends string>({ value, options, onChange, label }: { value: T; options: [T, string][]; onChange: (v: T) => void; label: string }) {
  return (
    <div className="inline-flex rounded-xl border border-line bg-surface p-1" role="radiogroup" aria-label={label}>
      {options.map(([v, text]) => (
        <button
          key={v}
          type="button"
          role="radio"
          aria-checked={value === v}
          onClick={() => onChange(v)}
          className={`rounded-lg px-3 py-1.5 text-xs font-semibold transition-colors ${value === v ? "bg-btn text-white" : "text-ink-2 hover:bg-surface-3"}`}
        >
          {text}
        </button>
      ))}
    </div>
  );
}

function ModelCard({ m, best, color }: { m: ModelEntry; best: boolean; color: string }) {
  const stats: [string, string][] = [
    ["Test accuracy", pct(m.test.accuracy)],
    ["Macro F1", pct(m.test.f1_macro)],
    ["Parameters", compact(m.params)],
    ["Model size (ONNX)", fmtMB(m.onnx_size_mb)],
    ["Inference (CPU)", fmtMs(m.onnx_latency_ms)],
    ["Training", `${m.epochs} ep · ${Math.max(1, Math.round(m.train_time_s / 60))} min`],
  ];
  return (
    <div className={`card p-6 ${best ? "ring-2 ring-brand/60" : ""}`}>
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="flex items-center gap-2 font-display text-lg font-bold text-ink">
            <span className="size-3 rounded-sm" style={{ background: color }} aria-hidden="true" />
            {m.name}
          </p>
          <p className="mt-1 text-sm text-ink-2">{m.architecture}</p>
        </div>
        {best && (
          <span className="chip shrink-0 border-brand/40 text-brand">
            <Award className="size-3.5" aria-hidden="true" /> Used on this site
          </span>
        )}
      </div>
      <dl className="mt-5 grid grid-cols-2 gap-x-4 gap-y-4 sm:grid-cols-3">
        {stats.map(([k, v]) => (
          <div key={k}>
            <dt className="text-xs text-muted">{k}</dt>
            <dd className="mt-0.5 font-display text-xl font-semibold text-ink">{v}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

function Tables({ metrics, models }: { metrics: MetricsFile; models: ModelEntry[] }) {
  const rows: [string, (m: ModelEntry) => string][] = [
    ["Test accuracy", (m) => pct(m.test.accuracy)],
    ["Precision (macro)", (m) => pct(m.test.precision_macro)],
    ["Recall (macro)", (m) => pct(m.test.recall_macro)],
    ["F1 (macro)", (m) => pct(m.test.f1_macro)],
    ["Best validation accuracy", (m) => pct(m.best_val_accuracy)],
    ["Parameters", (m) => m.params.toLocaleString("en-US")],
    ["Keras model size", (m) => fmtMB(m.keras_size_mb)],
    ["ONNX model size", (m) => fmtMB(m.onnx_size_mb)],
    ["Inference, TensorFlow CPU", (m) => fmtMs(m.keras_latency_ms)],
    ["Inference, ONNX Runtime CPU", (m) => fmtMs(m.onnx_latency_ms)],
    ["Epochs", (m) => String(m.epochs)],
    ["Training time", (m) => `${Math.round(m.train_time_s)} s`],
  ];
  return (
    <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-2">
      <div className="card overflow-x-auto p-6">
        <table className="w-full text-sm">
          <caption className="mb-3 text-left font-semibold text-ink">Summary (test split, n = {models[0]?.test.n})</caption>
          <thead>
            <tr className="border-b border-line text-left text-xs text-muted">
              <th className="py-2 pr-4 font-medium">Metric</th>
              {models.map((m) => (
                <th key={m.key} className="py-2 pr-4 font-medium">
                  {m.name}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map(([label, get]) => (
              <tr key={label} className="border-b border-line/60">
                <th className="py-2 pr-4 text-left font-normal text-ink-2">{label}</th>
                {models.map((m) => (
                  <td key={m.key} className="py-2 pr-4 text-ink tabular-nums">
                    {get(m)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="card overflow-x-auto p-6">
        <table className="w-full text-sm">
          <caption className="mb-3 text-left font-semibold text-ink">Per-class F1 score</caption>
          <thead>
            <tr className="border-b border-line text-left text-xs text-muted">
              <th className="py-2 pr-4 font-medium">Class</th>
              {models.map((m) => (
                <th key={m.key} className="py-2 pr-4 font-medium">
                  {m.name}
                </th>
              ))}
              <th className="py-2 pr-4 font-medium">Test images</th>
            </tr>
          </thead>
          <tbody>
            {metrics.class_names.map((c) => (
              <tr key={c} className="border-b border-line/60">
                <th className="py-2 pr-4 text-left font-normal text-ink-2">{metrics.display_names[c]}</th>
                {models.map((m) => (
                  <td key={m.key} className="py-2 pr-4 text-ink tabular-nums">
                    {pct(m.test.per_class[c]?.f1 ?? 0)}
                  </td>
                ))}
                <td className="py-2 pr-4 text-ink tabular-nums">{models[0]?.test.per_class[c]?.support ?? 0}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function ModelPerformance() {
  const { metrics } = useSiteData();
  const [metric, setMetric] = useState<"accuracy" | "loss">("accuracy");
  const [cmKey, setCmKey] = useState<string | null>(null);
  const [normalize, setNormalize] = useState(false);
  const [showTables, setShowTables] = useState(false);

  if (!metrics) {
    return (
      <section id="performance" className="border-t border-line py-20 sm:py-24">
        <div className="container-page">
          <SectionHeader eyebrow="Model performance" title="Custom CNN vs MobileNetV2">
            Results appear here after training (web/public/data/metrics.json).
          </SectionHeader>
        </div>
      </section>
    );
  }

  const models = ORDER.filter((k) => metrics.models[k]).map((k) => metrics.models[k]);
  const colorOf = (key: string) => `var(--chart-${ORDER.indexOf(key) + 1})`;
  const cmp = compareModels(metrics);
  const lossMax = Math.max(0.1, ...models.flatMap((m) => [...(m.history.loss ?? []), ...(m.history.val_loss ?? [])]));
  const cmModel = metrics.models[cmKey ?? metrics.best_model];

  return (
    <section id="performance" aria-labelledby="performance-title" className="border-t border-line py-20 sm:py-24">
      <div className="container-page">
        <SectionHeader eyebrow="Model performance" title="Custom CNN vs MobileNetV2" id="performance-title">
          Both models were trained on the same split and scored on the same {models[0]?.test.n} held-out test images that
          neither model saw during training.
        </SectionHeader>

        {metrics.demo && (
          <p className="mt-6 flex items-start gap-2 rounded-xl border border-[color-mix(in_oklab,var(--sev-low)_45%,transparent)] bg-[color-mix(in_oklab,var(--sev-low)_12%,transparent)] p-4 text-sm text-[var(--sev-low-ink)]">
            <TriangleAlert className="mt-0.5 size-4 shrink-0" aria-hidden="true" />
            These numbers come from the demo models (synthetic images, short training) and only show that the pipeline works.
            They will be replaced by the real results after training on the collected photos.
          </p>
        )}

        <div className="mt-10 grid grid-cols-1 gap-6 md:grid-cols-2">
          {models.map((m) => (
            <ModelCard key={m.key} m={m} best={m.key === metrics.best_model} color={colorOf(m.key)} />
          ))}
        </div>

        <div className="card mt-6 p-6">
          <h3 className="font-display text-lg font-semibold text-ink">Which model is better, and why?</h3>
          <p className="mt-2 leading-relaxed text-ink">{cmp.headline}</p>
          <ul className="mt-3 list-disc space-y-1.5 pl-5 text-sm leading-relaxed text-ink-2">
            {cmp.points.map((p) => (
              <li key={p}>{p}</li>
            ))}
          </ul>
          <p className="mt-3 text-xs text-muted">Selection rule: {metrics.selection_rule}.</p>
        </div>

        <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-2">
          <div className="card p-6">
            <h3 className="font-display text-lg font-semibold text-ink">Test-set scores</h3>
            <div className="mt-2 mb-4">
              <Legend items={models.map((m) => ({ label: m.name, color: colorOf(m.key) }))} />
            </div>
            <MetricBars models={models} colorOf={colorOf} />
          </div>
          <div className="card p-6">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <h3 className="font-display text-lg font-semibold text-ink">Confusion matrix</h3>
              <Toggle value={normalize ? "pct" : "count"} options={[["count", "Counts"], ["pct", "Percent"]]} onChange={(v) => setNormalize(v === "pct")} label="Cell values" />
            </div>
            <div className="mt-3 mb-4">
              <Toggle value={cmModel.key} options={models.map((m) => [m.key, m.name] as [string, string])} onChange={setCmKey} label="Model" />
            </div>
            <ConfusionMatrix matrix={cmModel.test.confusion_matrix} classNames={metrics.class_names} displayNames={metrics.display_names} normalize={normalize} />
          </div>
        </div>

        <div className="card mt-6 p-6">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h3 className="font-display text-lg font-semibold text-ink">Training and validation curves</h3>
            <Toggle value={metric} options={[["accuracy", "Accuracy"], ["loss", "Loss"]]} onChange={setMetric} label="Curve metric" />
          </div>
          <div className="mt-3">
            <Legend
              items={[
                { label: "Validation (solid)", color: "var(--ink-2)" },
                { label: "Training (faded)", color: "var(--ink-2)", faded: true },
              ]}
            />
          </div>
          <div className="mt-4 grid grid-cols-1 gap-8 md:grid-cols-2">
            {models.map((m) => (
              <div key={m.key}>
                <p className="mb-2 flex items-center gap-2 text-sm font-semibold text-ink">
                  <span className="size-2.5 rounded-sm" style={{ background: colorOf(m.key) }} aria-hidden="true" />
                  {m.name}
                </p>
                <TrainingCurve model={m} color={colorOf(m.key)} metric={metric} yMax={metric === "accuracy" ? 1 : lossMax} />
              </div>
            ))}
          </div>
          <p className="mt-3 text-xs text-muted">
            A growing gap between the faded training curve and the solid validation curve indicates overfitting. Early stopping keeps the
            weights from the best validation epoch.
          </p>
        </div>

        <button type="button" className="btn btn-secondary mt-6" onClick={() => setShowTables((s) => !s)} aria-expanded={showTables}>
          {showTables ? <ChartColumn className="size-4" aria-hidden="true" /> : <Table className="size-4" aria-hidden="true" />}
          {showTables ? "Hide data tables" : "Show data tables"}
        </button>
        {showTables && <Tables metrics={metrics} models={models} />}
      </div>
    </section>
  );
}
