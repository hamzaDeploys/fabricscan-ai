import { Check, CircleCheck, Copy, FileDown, ImageDown, RotateCcw, ScanSearch, Sparkles, TriangleAlert } from "lucide-react";
import { motion } from "motion/react";
import { useState, type ReactNode, type RefObject } from "react";

import type { AnalysisResult } from "../../hooks/useAnalyzer";
import { reportToText } from "../../lib/descriptionEngine";
import { copyText, downloadPdf, downloadPng } from "../../lib/exportReport";
import { areaText, capitalize, pct } from "../../lib/format";
import type { Zone } from "../../lib/localization";
import { ConfidenceBars } from "./ConfidenceBars";
import { SeverityBadge } from "./SeverityBadge";

const ZONES: Zone[] = [
  "upper-left", "upper-center", "upper-right",
  "middle-left", "center", "middle-right",
  "lower-left", "lower-center", "lower-right",
];

/** Tiny 3x3 map of the fabric with the defect's zone filled in. */
function ZoneGrid({ zone }: { zone: Zone }) {
  return (
    <span className="grid shrink-0 grid-cols-3 gap-[2px]" aria-hidden="true">
      {ZONES.map((z) => (
        <span
          key={z}
          className="size-[7px] rounded-[2px]"
          style={{ background: z === zone ? "var(--accent-bright)" : "color-mix(in oklab, var(--muted) 35%, transparent)" }}
        />
      ))}
    </span>
  );
}

/** Circular gauge of the top-class probability. */
function ConfidenceRing({ value, low }: { value: number; low: boolean }) {
  const r = 26;
  const c = 2 * Math.PI * r;
  return (
    <div className="relative size-[68px] shrink-0" role="img" aria-label={`Confidence ${pct(value)}`}>
      <svg viewBox="0 0 64 64" className="size-full -rotate-90">
        <circle cx="32" cy="32" r={r} fill="none" stroke="var(--surface-3)" strokeWidth="6" />
        <motion.circle
          cx="32"
          cy="32"
          r={r}
          fill="none"
          stroke={low ? "var(--sev-low)" : "var(--btn)"}
          strokeWidth="6"
          strokeLinecap="round"
          strokeDasharray={c}
          initial={{ strokeDashoffset: c }}
          animate={{ strokeDashoffset: c * (1 - value) }}
          transition={{ duration: 0.8, ease: "easeOut" }}
        />
      </svg>
      <span className="absolute inset-0 flex flex-col items-center justify-center leading-none">
        <span className="font-display text-base font-bold text-ink">{Math.round(value * 100)}%</span>
        <span className="mt-0.5 text-[9px] text-muted uppercase">sure</span>
      </span>
    </div>
  );
}

function Fact({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="min-w-0 px-3 py-3">
      <dt className="text-[11px] font-medium text-muted">{label}</dt>
      <dd className="mt-1.5 text-sm font-semibold text-ink">{children}</dd>
    </div>
  );
}

interface Props {
  result: AnalysisResult;
  canvasRef: RefObject<HTMLCanvasElement | null>;
  refining: { done: number; total: number } | null;
  onRefine: () => void;
  onReset: () => void;
}

export function ResultSummary({ result, canvasRef, refining, onRefine, onReset }: Props) {
  const { report } = result;
  const [status, setStatus] = useState<string | null>(null);
  const verdictInk = report.isDefect ? `var(--sev-${report.severity.toLowerCase()}-ink)` : "var(--sev-none-ink)";

  const act = async (done: string, fn: () => Promise<void>) => {
    try {
      await fn();
      setStatus(done);
    } catch (err) {
      setStatus(err instanceof Error ? err.message : "That did not work in this browser.");
    }
    setTimeout(() => setStatus(null), 2500);
  };

  return (
    <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }} data-testid="result">
      <div className="flex items-center gap-4">
        <ConfidenceRing value={report.confidence} low={report.lowConfidence} />
        <div className="min-w-0 flex-1">
          <p className="flex items-center gap-1.5 text-xs font-semibold tracking-wider uppercase" style={{ color: verdictInk }}>
            {report.isDefect ? <TriangleAlert className="size-3.5" aria-hidden="true" /> : <CircleCheck className="size-3.5" aria-hidden="true" />}
            {report.isDefect ? "Defect detected" : "No defect found"}
          </p>
          <h3 className="mt-1 font-display text-3xl leading-tight font-bold tracking-tight text-ink" data-testid="predicted-label">
            {report.label}
          </h3>
        </div>
        <button type="button" className="btn btn-ghost shrink-0 self-start px-2.5 py-2 text-xs" onClick={onReset} aria-label="New image" title="New image">
          <RotateCcw className="size-4" aria-hidden="true" />
          <span className="hidden sm:inline">New image</span>
        </button>
      </div>

      <dl className="mt-5 grid grid-cols-3 divide-x divide-line rounded-xl border border-line bg-surface">
        <Fact label="Severity">
          <SeverityBadge severity={report.severity} short />
        </Fact>
        <Fact label="Location">
          {report.zone ? (
            <span className="flex items-center gap-2">
              <ZoneGrid zone={report.zone as Zone} />
              <span className="leading-tight">{capitalize(report.zone.replace("-", " "))}</span>
            </span>
          ) : (
            <span className="text-muted">—</span>
          )}
        </Fact>
        <Fact label="Affected area">
          {report.areaPct !== null ? <span>≈ {areaText(report.areaPct)}</span> : <span className="text-muted">—</span>}
        </Fact>
      </dl>

      <div className="mt-5">
        <p className="flex items-center gap-1.5 text-[11px] font-semibold tracking-wider text-muted uppercase">
          <Sparkles className="size-3.5 text-accent" aria-hidden="true" /> Automatic description
        </p>
        <p className="mt-2 border-l-2 border-accent pl-3.5 text-[15px] leading-relaxed text-ink" data-testid="summary">
          {report.summary}
        </p>
        <p className={`mt-2 flex items-start gap-1.5 pl-4 text-xs ${report.lowConfidence ? "text-[var(--sev-low-ink)]" : "text-muted"}`}>
          {report.lowConfidence && <TriangleAlert className="size-3.5 shrink-0" aria-hidden="true" />}
          {report.confidenceNote}
        </p>
      </div>

      <div className="mt-6">
        <p className="mb-2.5 text-[11px] font-semibold tracking-wider text-muted uppercase">Confidence for every class</p>
        <ConfidenceBars ranked={report.ranked} />
      </div>

      {report.causes.length > 0 && (
        <details className="group mt-5 border-t border-line pt-4">
          <summary className="cursor-pointer text-sm font-semibold text-ink marker:text-muted">Possible causes ({report.causes.length})</summary>
          <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-ink-2">
            {report.causes.map((c) => (
              <li key={c}>{c}</li>
            ))}
          </ul>
        </details>
      )}

      <div className="mt-5 flex flex-wrap items-center gap-2 border-t border-line pt-4">
        <button type="button" className="btn btn-primary" onClick={() => act("PDF downloaded", () => downloadPdf(report, canvasRef.current!))}>
          <FileDown className="size-4" aria-hidden="true" /> PDF report
        </button>
        <button type="button" className="btn btn-secondary px-3" aria-label="Download PNG" onClick={() => act("PNG downloaded", () => downloadPng(report, canvasRef.current!))}>
          <ImageDown className="size-4" aria-hidden="true" /> PNG
        </button>
        <button type="button" className="btn btn-secondary px-3" aria-label="Copy report" onClick={() => act("Report copied", () => copyText(reportToText(report)))}>
          <Copy className="size-4" aria-hidden="true" /> Copy
        </button>
        {report.isDefect && !result.refined && (
          <button
            type="button"
            className="btn btn-secondary px-3"
            onClick={onRefine}
            disabled={refining !== null}
            title="Re-locate the defect with an occlusion map (49 extra model runs)"
          >
            <ScanSearch className="size-4" aria-hidden="true" />
            {refining ? `${refining.done}/${refining.total}` : "Refine"}
          </button>
        )}
      </div>
      <p className="mt-2 min-h-5 text-sm text-[var(--sev-none-ink)]" aria-live="polite">
        {status && (
          <span className="inline-flex items-center gap-1.5">
            <Check className="size-4" aria-hidden="true" /> {status}
          </span>
        )}
      </p>
    </motion.div>
  );
}
