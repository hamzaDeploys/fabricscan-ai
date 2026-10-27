import { Check, LoaderCircle } from "lucide-react";

import type { Phase } from "../../hooks/useAnalyzer";

const STEPS: { phase: Phase; label: string; detail: string }[] = [
  { phase: "preparing", label: "Preparing the image", detail: "Square crop and resize to 224 × 224" },
  { phase: "loading-model", label: "Loading the neural network", detail: "Only the first time, then cached" },
  { phase: "analyzing", label: "Classifying and locating", detail: "Then writing the report" },
];

export function AnalysisProgress({ phase }: { phase: Phase }) {
  const current = STEPS.findIndex((s) => s.phase === phase);
  return (
    <div className="flex h-full flex-col justify-center py-6">
      <p className="font-display text-xl font-semibold text-ink">Analysing…</p>
      <ol className="mt-6 space-y-5">
        {STEPS.map((s, i) => (
          <li key={s.phase} className="flex items-start gap-3">
            <span
              className={`mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-full text-xs font-semibold ${
                i < current ? "bg-[var(--sev-none)] text-white" : i === current ? "bg-btn text-white" : "bg-surface-3 text-muted"
              }`}
            >
              {i < current ? (
                <Check className="size-4" aria-hidden="true" />
              ) : i === current ? (
                <LoaderCircle className="size-4 animate-spin" aria-hidden="true" />
              ) : (
                i + 1
              )}
            </span>
            <span>
              <span className={`block text-sm font-semibold ${i <= current ? "text-ink" : "text-muted"}`}>{s.label}</span>
              <span className="block text-xs text-muted">{s.detail}</span>
            </span>
          </li>
        ))}
      </ol>
      <p className="sr-only" aria-live="polite">
        {STEPS[current]?.label}
      </p>
    </div>
  );
}
